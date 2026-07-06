from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from engine.board import Board
from engine.game import Game
from engine.types import Move, PlayerState

from .bots import ArenaBot, build_bot


@dataclass(frozen=True)
class ArenaConfig:
    width: int = 18
    height: int = 18
    max_turns: int = 900
    strategic_interval: int = 1


@dataclass
class MatchResult:
    seed: int
    winner: int | None
    turns: int
    half_turns: int
    finished: bool
    terminal_reason: str
    bot_names: list[str]
    total_moves: dict[int, int]
    active_half_turns: dict[int, int]
    zero_half_turns: dict[int, int]
    reverse_moves: dict[int, int]
    final_rankings: list[dict[str, Any]]
    dominance: dict[str, Any]

    @property
    def zero_half_turn_rate(self) -> float:
        denom = max(1, sum(self.active_half_turns.values()))
        return round(sum(self.zero_half_turns.values()) / denom, 4)

    @property
    def avg_moves_per_half_turn(self) -> float:
        denom = max(1, sum(self.active_half_turns.values()))
        return round(sum(self.total_moves.values()) / denom, 3)

    @property
    def reverse_move_rate(self) -> float:
        total = sum(self.total_moves.values())
        return round(sum(self.reverse_moves.values()) / total, 4) if total else 0.0


class ArenaRunner:
    def __init__(self, bots: list[ArenaBot], config: ArenaConfig | None = None):
        if len(bots) != 2:
            raise ValueError("ArenaRunner currently expects exactly two bots")
        self.bots = bots
        self.config = config or ArenaConfig()

    def run(self, seed: int) -> MatchResult:
        players = [
            PlayerState(index=idx, alive=True, name=self.bots[idx].name)
            for idx in range(2)
        ]
        board = Board.generate_map(self.config.width, self.config.height, 2, seed=seed)
        game = Game(board, players)
        for bot in self.bots:
            bot.reset(seed)

        half_turn = 0
        total_moves = {idx: 0 for idx in range(2)}
        active_half_turns = {idx: 0 for idx in range(2)}
        zero_half_turns = {idx: 0 for idx in range(2)}
        reverse_moves = {idx: 0 for idx in range(2)}
        previous_edges = {idx: set() for idx in range(2)}
        dominance = DominanceTracker()

        while not game.is_finished() and game.turn < self.config.max_turns:
            moves_by_player: dict[int, list[Move]] = {}
            for bot in self.bots:
                if not game.alive[bot.player_idx]:
                    continue
                view = game.get_player_view(bot.player_idx)
                moves = bot.moves(view, half_turn)
                moves_by_player[bot.player_idx] = moves
                active_half_turns[bot.player_idx] += 1
                total_moves[bot.player_idx] += len(moves)
                if not moves:
                    zero_half_turns[bot.player_idx] += 1
                reverse_moves[bot.player_idx] += _count_reverse_moves(
                    bot.player_idx,
                    moves,
                    previous_edges,
                )

            game.step(moves_by_player)
            half_turn += 1
            dominance.observe(game.turn, game.get_rankings())

        winner = _winner(game)
        finished = game.is_finished()
        return MatchResult(
            seed=seed,
            winner=winner,
            turns=game.turn,
            half_turns=half_turn,
            finished=finished,
            terminal_reason="general_captured" if finished else "max_turns",
            bot_names=[bot.name for bot in self.bots],
            total_moves=total_moves,
            active_half_turns=active_half_turns,
            zero_half_turns=zero_half_turns,
            reverse_moves=reverse_moves,
            final_rankings=game.get_rankings(),
            dominance=dominance.summary(game.turn),
        )


@dataclass
class DominanceTracker:
    army_ratio: float = 1.7
    tile_ratio: float = 1.4
    first_turn: int | None = None
    leader: int | None = None
    margin: float = 0.0

    def observe(self, turn: int, rankings: list[dict[str, Any]]) -> None:
        alive = [row for row in rankings if row.get("alive")]
        if len(alive) < 2:
            return
        leader = alive[0]
        runner_up = alive[1]
        army_ratio = leader["army"] / max(1, runner_up["army"])
        tile_ratio = leader["tiles"] / max(1, runner_up["tiles"])
        if army_ratio >= self.army_ratio and tile_ratio >= self.tile_ratio:
            if self.first_turn is None:
                self.first_turn = turn
                self.leader = leader["player"]
                self.margin = round(army_ratio, 2)
            elif self.leader == leader["player"]:
                self.margin = max(self.margin, round(army_ratio, 2))

    def summary(self, last_turn: int) -> dict[str, Any]:
        return {
            "leader": self.leader,
            "first_turn": self.first_turn,
            "age_turns": 0 if self.first_turn is None else max(0, last_turn - self.first_turn),
            "max_army_margin": self.margin,
        }


def run_batch(
    bot_specs: list[str],
    seeds: list[int],
    config: ArenaConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ArenaConfig()
    results: list[MatchResult] = []
    for seed in seeds:
        bots = [
            build_bot(spec, idx, cfg.strategic_interval)
            for idx, spec in enumerate(bot_specs)
        ]
        results.append(ArenaRunner(bots, cfg).run(seed))
    return summarize_results(results)


def summarize_results(results: list[MatchResult]) -> dict[str, Any]:
    if not results:
        return {"matches": 0, "results": []}

    bot_names = results[0].bot_names
    wins = {idx: 0 for idx in range(len(bot_names))}
    draws = 0
    for result in results:
        if result.winner is None:
            draws += 1
        else:
            wins[result.winner] += 1

    return {
        "matches": len(results),
        "bots": bot_names,
        "wins": wins,
        "draws": draws,
        "win_rates": {
            idx: round(wins[idx] / len(results), 4)
            for idx in wins
        },
        "avg_turns": round(mean(result.turns for result in results), 2),
        "finished_rate": round(sum(1 for result in results if result.finished) / len(results), 4),
        "avg_reverse_move_rate": round(mean(result.reverse_move_rate for result in results), 4),
        "avg_zero_half_turn_rate": round(mean(result.zero_half_turn_rate for result in results), 4),
        "avg_moves_per_half_turn": round(mean(result.avg_moves_per_half_turn for result in results), 3),
        "dominance_lag_turns": round(
            mean(result.dominance["age_turns"] for result in results if result.dominance["first_turn"] is not None),
            2,
        ) if any(result.dominance["first_turn"] is not None for result in results) else 0,
        "results": [result.__dict__ for result in results],
    }


def _count_reverse_moves(
    player_idx: int,
    moves: list[Move],
    previous_edges_by_player: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]],
) -> int:
    current_edges = {
        ((move.from_x, move.from_y), (move.to_x, move.to_y))
        for move in moves
    }
    previous_edges = previous_edges_by_player[player_idx]
    reverse_count = sum(
        1
        for edge in current_edges
        if (edge[1], edge[0]) in previous_edges
    )
    previous_edges_by_player[player_idx] = current_edges
    return reverse_count


def _winner(game: Game) -> int | None:
    alive = [idx for idx, is_alive in enumerate(game.alive) if is_alive]
    return alive[0] if len(alive) == 1 else None
