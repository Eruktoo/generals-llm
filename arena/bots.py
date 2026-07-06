from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agents import HeuristicAgent
from engine.random_bot import RandomBot
from engine.types import Move
from execution import Executor


class ArenaBot(Protocol):
    name: str
    player_idx: int

    def reset(self, seed: int) -> None:
        ...

    def moves(self, game_view: dict, half_turn: int) -> list[Move]:
        ...


@dataclass
class StrategyBot:
    player_idx: int
    personality: str = "baseline"
    strategic_interval: int = 1

    def __post_init__(self) -> None:
        self.name = f"heuristic:{self.personality}"
        self._agent = HeuristicAgent(self.player_idx, self.personality)
        self._strategy: dict | None = None

    def reset(self, seed: int) -> None:
        self._strategy = None

    def moves(self, game_view: dict, half_turn: int) -> list[Move]:
        if self._strategy is None or half_turn % max(1, self.strategic_interval) == 0:
            self._strategy = self._agent.decide(game_view)
        executor = Executor(
            self.player_idx,
            game_view["board"],
            self._strategy.get("constraints") or {},
        )
        return executor.execute(self._strategy)


@dataclass
class RandomMoveBot:
    player_idx: int

    def __post_init__(self) -> None:
        self.name = "random"
        self._bot = RandomBot(self.player_idx)

    def reset(self, seed: int) -> None:
        self._bot = RandomBot(self.player_idx, seed + self.player_idx * 10_007)

    def moves(self, game_view: dict, half_turn: int) -> list[Move]:
        return self._bot.request_move(game_view)


def build_bot(spec: str, player_idx: int, strategic_interval: int = 1) -> ArenaBot:
    normalized = spec.strip().lower()
    if normalized == "random":
        return RandomMoveBot(player_idx)
    if normalized.startswith("heuristic"):
        _, _, personality = normalized.partition(":")
        return StrategyBot(player_idx, personality or "baseline", strategic_interval)
    raise ValueError(f"unknown bot spec {spec!r}; use random or heuristic[:personality]")
