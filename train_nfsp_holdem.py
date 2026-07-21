#!/usr/bin/env python3
"""Headless NFSP training for six-player no-limit Texas Hold'em in OpenSpiel.

This is a research baseline, not a claim of professional-level poker strength.
It uses one shared agent for every seat (the information-state tensor contains
the seat id), a four-action no-limit abstraction, and Neural Fictitious
Self-Play (NFSP):

* a Double-DQN learns an approximate best response;
* a policy network learns the average strategy from a reservoir buffer;
* an anticipatory mixture generates self-play experience.

Example:
    python train_nfsp_holdem.py --episodes 2000000 --device cuda
    python train_nfsp_holdem.py --resume checkpoints/latest.pt --episodes 500000
    python train_nfsp_holdem.py --eval-only checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, NamedTuple, Optional

import numpy as np
import pyspiel
import torch
from torch import nn
from torch.nn import functional as F


LOG = logging.getLogger("nfsp_holdem")


@dataclass
class Config:
    episodes: int = 2_000_000
    stack: int = 10_000
    small_blind: int = 50
    big_blind: int = 100
    hidden_size: int = 256
    hidden_layers: int = 3
    batch_size: int = 256
    replay_capacity: int = 1_000_000
    reservoir_capacity: int = 2_000_000
    replay_warmup: int = 20_000
    gamma: float = 1.0
    learning_rate: float = 1e-4
    anticipatory: float = 0.1
    epsilon_start: float = 0.12
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 2_000_000
    target_update: int = 10_000
    train_every: int = 4
    sl_train_every: int = 4
    checkpoint_every: int = 50_000
    evaluate_every: int = 25_000
    eval_hands: int = 2_000
    log_every: int = 1_000
    seed: int = 7


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_legal_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.data.append(transition)

    def sample(self, n: int) -> list[Transition]:
        return random.sample(self.data, n)

    def __len__(self) -> int:
        return len(self.data)


class ReservoirBuffer:
    """Uniform sample of the entire stream, as required by NFSP."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: list[tuple[np.ndarray, int, np.ndarray]] = []
        self.seen = 0

    def add(self, state: np.ndarray, action: int, legal_mask: np.ndarray) -> None:
        self.seen += 1
        item = (state, action, legal_mask)
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            index = random.randrange(self.seen)
            if index < self.capacity:
                self.data[index] = item

    def sample(self, n: int) -> list[tuple[np.ndarray, int, np.ndarray]]:
        return random.sample(self.data, n)

    def __len__(self) -> int:
        return len(self.data)


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, width: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        size = input_size
        for _ in range(depth):
            layers.extend((nn.Linear(size, width), nn.ReLU()))
            size = width
        layers.append(nn.Linear(size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_game(cfg: Config) -> pyspiel.Game:
    """Create six-max NLHE with standard cards and an actionable abstraction."""
    params = {
        "numPlayers": 6,
        "betting": "nolimit",
        "stack": " ".join([str(cfg.stack)] * 6),
        "blind": f"{cfg.small_blind} {cfg.big_blind} 0 0 0 0",
        "numRounds": 4,
        "firstPlayer": "2 0 0 0",
        "numSuits": 4,
        "numRanks": 13,
        "numHoleCards": 2,
        "numBoardCards": "0 3 1 1",
        # fold, check/call, pot-sized raise, all-in
        "bettingAbstraction": "fcpa",
    }
    game = pyspiel.load_game("universal_poker", params)
    if game.num_players() != 6:
        raise RuntimeError(f"Expected 6 players, got {game.num_players()}")
    return game


def legal_mask(state: pyspiel.State, num_actions: int) -> np.ndarray:
    mask = np.zeros(num_actions, dtype=np.bool_)
    mask[state.legal_actions()] = True
    return mask


def sample_chance(state: pyspiel.State, rng: np.random.Generator) -> int:
    actions, probabilities = zip(*state.chance_outcomes())
    return int(rng.choice(actions, p=probabilities))


class NFSPAgent:
    def __init__(self, info_size: int, num_actions: int, cfg: Config, device: torch.device):
        self.cfg, self.device = cfg, device
        self.num_actions = num_actions
        args = (info_size, num_actions, cfg.hidden_size, cfg.hidden_layers)
        self.q = MLP(*args).to(device)
        self.target_q = MLP(*args).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.average_policy = MLP(*args).to(device)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=cfg.learning_rate)
        self.policy_optimizer = torch.optim.Adam(
            self.average_policy.parameters(), lr=cfg.learning_rate
        )
        self.replay = ReplayBuffer(cfg.replay_capacity)
        self.reservoir = ReservoirBuffer(cfg.reservoir_capacity)
        self.environment_steps = 0
        self.gradient_steps = 0

    def epsilon(self) -> float:
        fraction = min(1.0, self.environment_steps / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + fraction * (
            self.cfg.epsilon_end - self.cfg.epsilon_start
        )

    @torch.inference_mode()
    def act_best_response(self, info: np.ndarray, legal: np.ndarray) -> int:
        choices = np.flatnonzero(legal)
        if random.random() < self.epsilon():
            return int(random.choice(choices))
        values = self.q(torch.as_tensor(info, device=self.device).unsqueeze(0))[0]
        values = values.masked_fill(
            ~torch.as_tensor(legal, device=self.device), -torch.inf
        )
        return int(values.argmax().item())

    @torch.inference_mode()
    def act_average(self, info: np.ndarray, legal: np.ndarray, stochastic: bool = True) -> int:
        logits = self.average_policy(
            torch.as_tensor(info, device=self.device).unsqueeze(0)
        )[0]
        logits = logits.masked_fill(
            ~torch.as_tensor(legal, device=self.device), -torch.inf
        )
        if not stochastic:
            return int(logits.argmax().item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())

    def maybe_train(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if len(self.replay) >= max(self.cfg.replay_warmup, self.cfg.batch_size):
            if self.environment_steps % self.cfg.train_every == 0:
                metrics["q_loss"] = self._train_q()
        if len(self.reservoir) >= self.cfg.batch_size:
            if self.environment_steps % self.cfg.sl_train_every == 0:
                metrics["policy_loss"] = self._train_policy()
        return metrics

    def _train_q(self) -> float:
        batch = self.replay.sample(self.cfg.batch_size)
        states = torch.as_tensor(np.stack([x.state for x in batch]), device=self.device)
        actions = torch.as_tensor([x.action for x in batch], device=self.device)
        rewards = torch.as_tensor([x.reward for x in batch], device=self.device)
        next_states = torch.as_tensor(
            np.stack([x.next_state for x in batch]), device=self.device
        )
        dones = torch.as_tensor([x.done for x in batch], device=self.device)
        masks = torch.as_tensor(
            np.stack([x.next_legal_mask for x in batch]), device=self.device
        )

        predicted = self.q(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            # Double-DQN: online network selects; target network evaluates.
            online_next = self.q(next_states).masked_fill(~masks, -torch.inf)
            next_actions = online_next.argmax(dim=1)
            next_values = self.target_q(next_states).gather(
                1, next_actions[:, None]
            ).squeeze(1)
            next_values = torch.where(dones, torch.zeros_like(next_values), next_values)
            target = rewards + self.cfg.gamma * next_values
        loss = F.smooth_l1_loss(predicted, target)
        self.q_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.q_optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.cfg.target_update == 0:
            self.target_q.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def _train_policy(self) -> float:
        batch = self.reservoir.sample(self.cfg.batch_size)
        states = torch.as_tensor(np.stack([x[0] for x in batch]), device=self.device)
        actions = torch.as_tensor([x[1] for x in batch], device=self.device)
        masks = torch.as_tensor(np.stack([x[2] for x in batch]), device=self.device)
        logits = self.average_policy(states).masked_fill(~masks, -1e9)
        loss = F.cross_entropy(logits, actions)
        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.average_policy.parameters(), 10.0)
        self.policy_optimizer.step()
        return float(loss.item())

    def checkpoint(self) -> dict:
        return {
            "q": self.q.state_dict(),
            "target_q": self.target_q.state_dict(),
            "average_policy": self.average_policy.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
        }

    def restore(self, data: dict) -> None:
        self.q.load_state_dict(data["q"])
        self.target_q.load_state_dict(data.get("target_q", data["q"]))
        self.average_policy.load_state_dict(data["average_policy"])
        if "q_optimizer" in data:
            self.q_optimizer.load_state_dict(data["q_optimizer"])
            self.policy_optimizer.load_state_dict(data["policy_optimizer"])
        self.environment_steps = data.get("environment_steps", 0)
        self.gradient_steps = data.get("gradient_steps", 0)


def play_training_hand(
    game: pyspiel.Game,
    agent: NFSPAgent,
    rng: np.random.Generator,
) -> np.ndarray:
    state = game.new_initial_state()
    num_actions = game.num_distinct_actions()
    # NFSP chooses each player's policy mode for the whole episode.
    best_response_mode = rng.random(game.num_players()) < agent.cfg.anticipatory
    pending: list[Optional[tuple[np.ndarray, int]]] = [None] * game.num_players()

    while not state.is_terminal():
        if state.is_chance_node():
            state.apply_action(sample_chance(state, rng))
            continue

        player = state.current_player()
        info = np.asarray(state.information_state_tensor(player), dtype=np.float32)
        mask = legal_mask(state, num_actions)

        # The reward between two decisions is zero in terminal-reward poker.
        if pending[player] is not None:
            old_info, old_action = pending[player]
            agent.replay.add(Transition(old_info, old_action, 0.0, info, False, mask))

        if best_response_mode[player]:
            action = agent.act_best_response(info, mask)
            agent.reservoir.add(info, action, mask)
        else:
            action = agent.act_average(info, mask)
        pending[player] = (info, action)
        state.apply_action(action)
        agent.environment_steps += 1
        agent.maybe_train()

    returns = np.asarray(state.returns(), dtype=np.float32)
    zero_info = np.zeros(game.information_state_tensor_shape()[0], dtype=np.float32)
    zero_mask = np.zeros(num_actions, dtype=np.bool_)
    for player, item in enumerate(pending):
        if item is not None:
            info, action = item
            agent.replay.add(
                Transition(info, action, float(returns[player]), zero_info, True, zero_mask)
            )
    return returns


@torch.inference_mode()
def evaluate(
    game: pyspiel.Game,
    agent: NFSPAgent,
    hands: int,
    seed: int,
) -> dict[str, float]:
    """Rotate the learned average-policy bot through seats vs five random bots."""
    rng = np.random.default_rng(seed)
    profit = 0.0
    wins = ties = 0
    for hand in range(hands):
        hero = hand % game.num_players()
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                action = sample_chance(state, rng)
            else:
                player = state.current_player()
                mask = legal_mask(state, game.num_distinct_actions())
                if player == hero:
                    info = np.asarray(
                        state.information_state_tensor(player), dtype=np.float32
                    )
                    action = agent.act_average(info, mask, stochastic=True)
                else:
                    action = int(rng.choice(np.flatnonzero(mask)))
            state.apply_action(action)
        result = float(state.returns()[hero])
        profit += result
        wins += result > 0
        ties += result == 0
    return {
        "hands": hands,
        "win_rate": wins / hands,
        "non_loss_rate": (wins + ties) / hands,
        "mean_profit_chips": profit / hands,
        "mean_profit_big_blinds": profit / hands / agent.cfg.big_blind,
    }


def save_checkpoint(path: Path, agent: NFSPAgent, cfg: Config, episode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "episode": episode,
        "config": asdict(cfg),
        "agent": agent.checkpoint(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)  # Atomic on a single filesystem.


def load_checkpoint(path: Path, agent: NFSPAgent, restore_rng: bool = True) -> int:
    payload = torch.load(path, map_location=agent.device, weights_only=False)
    agent.restore(payload["agent"])
    if restore_rng and "rng" in payload:
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"])
    return int(payload.get("episode", 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=Config.episodes)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--eval-hands", type=int, default=Config.eval_hands)
    parser.add_argument("--checkpoint-every", type=int, default=Config.checkpoint_every)
    parser.add_argument("--evaluate-every", type=int, default=Config.evaluate_every)
    parser.add_argument("--seed", type=int, default=Config.seed)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    cfg = Config(
        episodes=args.episodes,
        eval_hands=args.eval_hands,
        checkpoint_every=args.checkpoint_every,
        evaluate_every=args.evaluate_every,
        seed=args.seed,
    )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # Small MLP calls often run faster without excessive CPU thread fan-out.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = choose_device(args.device)
    game = make_game(cfg)
    info_size = game.information_state_tensor_shape()[0]
    agent = NFSPAgent(info_size, game.num_distinct_actions(), cfg, device)
    LOG.info(
        "game=%s players=%d info=%d actions=%d device=%s",
        game.get_type().short_name,
        game.num_players(),
        info_size,
        game.num_distinct_actions(),
        device,
    )

    if args.eval_only:
        episode = load_checkpoint(args.eval_only, agent, restore_rng=False)
        metrics = evaluate(game, agent, cfg.eval_hands, cfg.seed + 1)
        print(json.dumps({"episode": episode, **metrics}, indent=2))
        return

    start_episode = load_checkpoint(args.resume, agent) if args.resume else 0
    rng = np.random.default_rng(cfg.seed + start_episode)
    window: Deque[float] = deque(maxlen=cfg.log_every)
    started = time.monotonic()
    last_episode = start_episode
    try:
        for episode in range(start_episode + 1, start_episode + cfg.episodes + 1):
            last_episode = episode
            returns = play_training_hand(game, agent, rng)
            window.append(float(np.mean(np.abs(returns))))
            if episode % cfg.log_every == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                LOG.info(
                    "episode=%d hands/s=%.2f steps=%d replay=%d reservoir=%d "
                    "epsilon=%.4f mean_abs_return=%.1f",
                    episode,
                    (episode - start_episode) / elapsed,
                    agent.environment_steps,
                    len(agent.replay),
                    len(agent.reservoir),
                    agent.epsilon(),
                    float(np.mean(window)),
                )
            if cfg.evaluate_every and episode % cfg.evaluate_every == 0:
                LOG.info("evaluation=%s", json.dumps(evaluate(
                    game, agent, cfg.eval_hands, cfg.seed + episode
                ), sort_keys=True))
            if cfg.checkpoint_every and episode % cfg.checkpoint_every == 0:
                numbered = args.checkpoint_dir / f"nfsp_episode_{episode:09d}.pt"
                save_checkpoint(numbered, agent, cfg, episode)
                save_checkpoint(args.checkpoint_dir / "latest.pt", agent, cfg, episode)
                LOG.info("saved checkpoint %s", numbered)
    except KeyboardInterrupt:
        LOG.warning("Interrupted; saving a recovery checkpoint")
    finally:
        final_path = args.checkpoint_dir / "latest.pt"
        save_checkpoint(final_path, agent, cfg, last_episode)
        LOG.info("final checkpoint=%s episode=%d", final_path, last_episode)


if __name__ == "__main__":
    main()
