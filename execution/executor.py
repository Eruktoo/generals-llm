from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from engine.board import TILE_FOG, TILE_FOG_OBSTACLE, Board
from engine.game import Game
from engine.types import Move, Tile, TileType


DEFAULT_CONSTRAINTS = {
    "min_general_garrison": 20,
    "min_city_garrison": 5,
    "max_commitment_percent": 50,
    "avoid_fog": False,
}


@dataclass(frozen=True)
class Region:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class ExecutionDiagnostic:
    objective_type: str
    status: str
    reason: str
    generated_moves: int = 0


@dataclass
class ExecutionResult:
    moves: list[Move]
    diagnostics: list[ExecutionDiagnostic] = field(default_factory=list)


class Executor:
    def __init__(
        self,
        player_idx: int,
        board_view: Board,
        constraints: dict | None = None,
    ):
        self.player_idx = player_idx
        self.board = board_view
        self.base_constraints = {**DEFAULT_CONSTRAINTS, **(constraints or {})}
        self._validator = Game()
        self._validator.board = board_view

    def execute(self, strategy: dict) -> list[Move]:
        return self.execute_with_diagnostics(strategy).moves

    def execute_with_diagnostics(self, strategy: dict) -> ExecutionResult:
        constraints = {
            **self.base_constraints,
            **(strategy.get("constraints") or {}),
        }
        used_sources: set[tuple[int, int]] = set()
        moves: list[Move] = []
        diagnostics: list[ExecutionDiagnostic] = []

        decapitation_moves = self._decapitation_moves(used_sources)
        if decapitation_moves:
            moves.extend(decapitation_moves)
            for move in decapitation_moves:
                used_sources.add((move.from_x, move.from_y))
            diagnostics.append(
                ExecutionDiagnostic(
                    "tactical_decapitation",
                    "ok",
                    "visible_enemy_general_override",
                    len(decapitation_moves),
                )
            )
        else:
            diagnostics.append(
                ExecutionDiagnostic(
                    "tactical_decapitation",
                    "blocked",
                    "no_visible_reachable_enemy_general",
                    0,
                )
            )

        endgame_hunt_moves = self._endgame_hunt_moves(strategy, constraints, used_sources)
        if endgame_hunt_moves:
            moves.extend(endgame_hunt_moves)
            for move in endgame_hunt_moves:
                used_sources.add((move.from_x, move.from_y))
            diagnostics.append(
                ExecutionDiagnostic(
                    "endgame_hunt",
                    "ok",
                    "superiority_push_visible_enemy_or_fog",
                    len(endgame_hunt_moves),
                )
            )
        elif self._is_endgame_superior(strategy):
            diagnostics.append(
                ExecutionDiagnostic(
                    "endgame_hunt",
                    "blocked",
                    "no_frontier_or_movable_stack",
                    0,
                )
            )

        for order in strategy.get("direct_orders") or []:
            move = self._move_from_direct_order(order, constraints, used_sources)
            if move is not None:
                moves.append(move)
                used_sources.add((move.from_x, move.from_y))
                diagnostics.append(ExecutionDiagnostic("direct_order", "ok", "accepted", 1))
            else:
                diagnostics.append(ExecutionDiagnostic("direct_order", "blocked", "invalid_or_constrained", 0))

        objectives = sorted(
            strategy.get("objectives") or [],
            key=lambda objective: float(objective.get("priority", 0)),
            reverse=True,
        )
        for objective in objectives:
            objective_moves = self._moves_for_objective(objective, constraints, used_sources)
            diagnostics.append(
                ExecutionDiagnostic(
                    str(objective.get("type", "unknown")),
                    "ok" if objective_moves else "blocked",
                    "generated" if objective_moves else self._blocked_reason(objective, constraints, used_sources),
                    len(objective_moves),
                )
            )
            for move in objective_moves:
                moves.append(move)
                used_sources.add((move.from_x, move.from_y))

        if not moves:
            fallback_moves = self._safe_expansion_fallback(used_sources)
            diagnostics.append(
                ExecutionDiagnostic(
                    "fallback_expand",
                    "ok" if fallback_moves else "blocked",
                    "relaxed_safe_expansion" if fallback_moves else "no_safe_adjacent_source",
                    len(fallback_moves),
                )
            )
            moves.extend(fallback_moves)

        return ExecutionResult(moves, diagnostics)

    def _moves_for_objective(
        self,
        objective: dict,
        constraints: dict,
        used_sources: set[tuple[int, int]],
    ) -> list[Move]:
        objective_type = objective.get("type")
        if objective_type == "expand_region":
            return self._expand_region(objective, constraints, used_sources)
        if objective_type == "reinforce_region":
            return self._reinforce_region(objective, constraints, used_sources)
        if objective_type == "attack_position":
            return self._attack_position(objective, constraints, used_sources)
        if objective_type == "defend_region":
            return self._defend_region(objective, constraints, used_sources)
        return []

    def _expand_region(
        self,
        objective: dict,
        constraints: dict,
        used_sources: set[tuple[int, int]],
    ) -> list[Move]:
        region = self._parse_region(objective.get("region"))
        if region is None:
            return []

        center = region.center
        candidates: list[tuple[float, int, int, int, int]] = []
        for x, y, tile in self._own_tiles(region):
            if (x, y) in used_sources or tile.army <= 1:
                continue
            for nx, ny in self._neighbors(x, y):
                if not region.contains(nx, ny):
                    continue
                target = self.board.tiles[ny][nx]
                if not self._is_expand_target(target):
                    continue
                if not self._destination_allowed(nx, ny, constraints):
                    continue
                candidates.append(
                    (
                        self._distance_to_point(nx, ny, center),
                        -tile.army,
                        x,
                        y,
                        self._pack(nx, ny),
                    )
                )

        moves: list[Move] = []
        for _, _, x, y, packed_dest in sorted(candidates):
            if (x, y) in used_sources or any((m.from_x, m.from_y) == (x, y) for m in moves):
                continue
            nx, ny = self._unpack(packed_dest)
            move = self._build_constrained_move(x, y, nx, ny, constraints)
            if move is not None:
                moves.append(move)
        return moves

    def _reinforce_region(
        self,
        objective: dict,
        constraints: dict,
        used_sources: set[tuple[int, int]],
    ) -> list[Move]:
        region = self._parse_region(objective.get("region"))
        if region is None:
            return []

        weak_tiles = self._weak_tiles_in_region(region)
        if not weak_tiles:
            return []

        sources = [
            (x, y, tile)
            for x, y, tile in self._own_tiles()
            if not region.contains(x, y) and (x, y) not in used_sources and tile.army > 1
        ]
        sources.sort(key=lambda item: (-item[2].army, self._nearest_distance(item[0], item[1], weak_tiles)))

        moves: list[Move] = []
        reserved_targets: set[tuple[int, int]] = set()
        for x, y, _ in sources:
            target_x, target_y, _ = min(
                weak_tiles,
                key=lambda item: (self._manhattan(x, y, item[0], item[1]), item[2].army),
            )
            if (target_x, target_y) in reserved_targets and len(reserved_targets) < len(weak_tiles):
                continue
            step = self._best_step_toward(x, y, target_x, target_y, constraints, prefer_owned=True)
            if step is None:
                continue
            nx, ny = step
            move = self._build_constrained_move(x, y, nx, ny, constraints)
            if move is not None:
                moves.append(move)
                reserved_targets.add((target_x, target_y))
        return moves

    def _attack_position(
        self,
        objective: dict,
        constraints: dict,
        used_sources: set[tuple[int, int]],
    ) -> list[Move]:
        target = objective.get("target") or {}
        if not self._is_int(target.get("x")) or not self._is_int(target.get("y")):
            return []
        target_x = int(target["x"])
        target_y = int(target["y"])
        if not self.board.in_bounds(target_x, target_y):
            return []

        commitment = objective.get("commitment", "limited")
        max_stacks = 4 if commitment == "full" else 2
        target_tile = self.board.tiles[target_y][target_x]
        raw_sources = [
            (x, y, tile)
            for x, y, tile in self._own_tiles()
            if (x, y) not in used_sources and tile.army > 1
        ]
        raw_sources.sort(
            key=lambda item: (
                self._manhattan(item[0], item[1], target_x, target_y),
                -item[2].army,
                item[1],
                item[0],
            )
        )
        raw_sources = raw_sources[: max(5, max_stacks * 2)]
        source_plans = [
            (x, y, tile, path)
            for x, y, tile in raw_sources
            for path in [self._path_to(x, y, target_x, target_y, constraints, prefer_owned=False)]
            if path is not None and len(path) > 1
        ]
        source_plans.sort(key=lambda item: (len(item[3]), -item[2].army, item[1], item[0]))
        if not source_plans:
            return []

        if self._should_stage_attack(target_tile):
            available = self._available_attack_force(source_plans, max_distance=4)
            required = self._required_attack_force(target_tile)
            if available < required:
                staged = self._stage_attack(source_plans, target_x, target_y, constraints, max_stacks)
                if staged:
                    return staged

        moves: list[Move] = []
        reserved_destinations: set[tuple[int, int]] = set()
        for x, y, tile, path in source_plans[:max_stacks]:
            nx, ny = path[1]
            if (nx, ny) in reserved_destinations:
                continue
            take_half = self._attack_take_half(tile, self.board.tiles[ny][nx], target_tile)
            move = self._build_constrained_move(x, y, nx, ny, constraints, take_half)
            if move is None:
                move = self._build_constrained_move(x, y, nx, ny, constraints)
            if move is not None:
                moves.append(move)
                reserved_destinations.add((nx, ny))
        return moves

    def _should_stage_attack(self, target: Tile) -> bool:
        if target.type == TileType.GENERAL and target.occupier != self.player_idx:
            return False
        return target.occupier != self.player_idx and (
            target.type == TileType.CITY
            or target.occupier == TILE_FOG
            or (target.occupier is not None and target.occupier >= 0)
        )

    def _available_attack_force(
        self,
        source_plans: list[tuple[int, int, Tile, list[tuple[int, int]]]],
        max_distance: int,
    ) -> int:
        return sum(
            max(0, self._moving_army(tile.army, False))
            for _, _, tile, path in source_plans
            if len(path) - 1 <= max_distance
        )

    def _required_attack_force(self, target: Tile) -> int:
        if target.occupier == TILE_FOG:
            return 25
        if target.type == TileType.CITY:
            return max(8, int(target.army * 1.5) + 1)
        if target.occupier is not None and target.occupier >= 0:
            return max(4, int(target.army * 1.3) + 1)
        return max(2, target.army + 1)

    def _stage_attack(
        self,
        source_plans: list[tuple[int, int, Tile, list[tuple[int, int]]]],
        target_x: int,
        target_y: int,
        constraints: dict,
        max_stacks: int,
    ) -> list[Move]:
        staging = min(
            source_plans,
            key=lambda item: (
                len(item[3]),
                -item[2].army,
                self._frontier_pressure(item[0], item[1]),
                item[1],
                item[0],
            ),
        )
        staging_pos = (staging[0], staging[1])
        merge_sources = [
            item for item in source_plans
            if (item[0], item[1]) != staging_pos
        ]
        merge_sources.sort(key=lambda item: (-item[2].army, len(item[3]), item[1], item[0]))

        moves: list[Move] = []
        reserved_destinations: set[tuple[int, int]] = set()
        for x, y, _tile, _path in merge_sources[:max_stacks]:
            path = self._path_to(x, y, staging_pos[0], staging_pos[1], constraints, prefer_owned=True)
            if path is None or len(path) < 2:
                continue
            nx, ny = path[1]
            if (nx, ny) in reserved_destinations:
                continue
            move = self._build_constrained_move(x, y, nx, ny, constraints, take_half=True)
            if move is None:
                move = self._build_constrained_move(x, y, nx, ny, constraints)
            if move is not None:
                moves.append(move)
                reserved_destinations.add((nx, ny))

        if moves:
            return moves

        x, y, tile, path = staging
        if len(path) < 2:
            return []
        nx, ny = path[1]
        if self._manhattan(nx, ny, target_x, target_y) >= self._manhattan(x, y, target_x, target_y):
            return []
        move = self._build_constrained_move(x, y, nx, ny, constraints, take_half=True)
        if move is None and tile.army >= self._required_attack_force(self.board.tiles[target_y][target_x]):
            move = self._build_constrained_move(x, y, nx, ny, constraints)
        return [move] if move is not None else []

    def _attack_take_half(self, source: Tile, destination: Tile, final_target: Tile) -> bool | None:
        if destination.occupier == self.player_idx:
            return True
        if final_target.type == TileType.GENERAL and final_target.occupier != self.player_idx:
            return False
        if destination.type == TileType.CITY and destination.occupier != self.player_idx:
            return True if source.army // 2 > destination.army else False
        if destination.occupier == TILE_FOG:
            return True
        return None

    def _frontier_pressure(self, x: int, y: int) -> int:
        pressure = 0
        for nx, ny in self._neighbors(x, y):
            tile = self.board.tiles[ny][nx]
            if tile.occupier is not None and tile.occupier >= 0 and tile.occupier != self.player_idx:
                pressure += tile.army
        return pressure

    def _defend_region(
        self,
        objective: dict,
        constraints: dict,
        used_sources: set[tuple[int, int]],
    ) -> list[Move]:
        region = self._parse_region(objective.get("region"))
        if region is None:
            return []

        border = [
            (x, y, tile)
            for x, y, tile in self._own_tiles(region)
            if self._is_border_tile(x, y, region)
        ]
        deep_sources = [
            (x, y, tile)
            for x, y, tile in self._own_tiles(region)
            if not self._is_border_tile(x, y, region) and (x, y) not in used_sources and tile.army > 1
        ]
        if not border or not deep_sources:
            return []

        border.sort(key=lambda item: (item[2].army, item[0], item[1]))
        deep_sources.sort(key=lambda item: (-item[2].army, item[0], item[1]))

        moves: list[Move] = []
        for x, y, _ in deep_sources:
            target_x, target_y, _ = min(
                border,
                key=lambda item: (item[2].army, self._manhattan(x, y, item[0], item[1])),
            )
            step = self._best_step_toward(x, y, target_x, target_y, constraints, prefer_owned=True)
            if step is None:
                continue
            nx, ny = step
            move = self._build_constrained_move(x, y, nx, ny, constraints)
            if move is not None:
                moves.append(move)
        return moves

    def _move_from_direct_order(
        self,
        order: dict,
        constraints: dict,
        used_sources: set[tuple[int, int]],
    ) -> Move | None:
        source = order.get("from") or {}
        target = order.get("to") or {}
        if not all(self._is_int(value) for value in (source.get("x"), source.get("y"), target.get("x"), target.get("y"))):
            return None

        from_x = int(source["x"])
        from_y = int(source["y"])
        to_x = int(target["x"])
        to_y = int(target["y"])
        if (from_x, from_y) in used_sources:
            return None
        if not self.board.in_bounds(from_x, from_y):
            return None
        source_tile = self.board.tiles[from_y][from_x]
        requested_amount = order.get("amount")
        if self._is_int(requested_amount) and int(requested_amount) <= source_tile.army // 2:
            preferred_modes = [True, False]
        else:
            preferred_modes = [False, True]

        for take_half in preferred_modes:
            move = self._build_constrained_move(from_x, from_y, to_x, to_y, constraints, take_half)
            if move is not None:
                return move
        return None

    def _build_constrained_move(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        constraints: dict,
        take_half: bool | None = None,
    ) -> Move | None:
        if not self.board.in_bounds(from_x, from_y) or not self.board.in_bounds(to_x, to_y):
            return None
        if not self._destination_allowed(to_x, to_y, constraints):
            return None

        source = self.board.tiles[from_y][from_x]
        possible = [take_half] if take_half is not None else [False, True]
        for half in possible:
            if half is None:
                continue
            move = Move(from_x, from_y, to_x, to_y, half)
            if not self._validator._valid_move(self.player_idx, move):
                continue
            moving_army = self._moving_army(source.army, half)
            if moving_army <= 0:
                continue
            if not self._respects_source_constraints(source, moving_army, constraints):
                continue
            return move

        # Fallback: constraints too strict, retry with minimal garrisons and full local commitment.
        relaxed = {
            **constraints,
            "min_general_garrison": 1,
            "min_city_garrison": 1,
            "max_commitment_percent": 100,
        }
        for half in possible:
            if half is None:
                continue
            move = Move(from_x, from_y, to_x, to_y, half)
            if not self._validator._valid_move(self.player_idx, move):
                continue
            moving_army = self._moving_army(source.army, half)
            if moving_army <= 0:
                continue
            if not self._respects_source_constraints(source, moving_army, relaxed):
                continue
            return move
        return None

    def _respects_source_constraints(self, source: Tile, moving_army: int, constraints: dict) -> bool:
        total_army = max(1, self._total_visible_army())
        max_commitment = int(total_army * float(constraints.get("max_commitment_percent", 100)) / 100)
        if moving_army > max_commitment:
            return False

        remaining = source.army - moving_army
        if source.type == TileType.GENERAL:
            return remaining >= int(constraints.get("min_general_garrison", 0))
        if source.type == TileType.CITY:
            return remaining >= int(constraints.get("min_city_garrison", 0))
        return remaining >= 1

    def _destination_allowed(self, x: int, y: int, constraints: dict) -> bool:
        tile = self.board.tiles[y][x]
        if tile.type == TileType.MOUNTAIN or tile.occupier == TILE_FOG_OBSTACLE:
            return False
        if constraints.get("avoid_fog") and tile.occupier == TILE_FOG:
            return False
        return True

    def _decapitation_moves(self, used_sources: set[tuple[int, int]]) -> list[Move]:
        targets = self._visible_enemy_generals()
        if not targets:
            return []

        relaxed = {
            **self.base_constraints,
            "min_general_garrison": 1,
            "min_city_garrison": 1,
            "max_commitment_percent": 100,
            "avoid_fog": False,
        }
        moves: list[Move] = []
        reserved_sources = set(used_sources)

        for target_x, target_y, target_tile in targets:
            direct = self._direct_general_attack(target_x, target_y, target_tile, relaxed, reserved_sources)
            if direct is not None:
                moves.append(direct)
                reserved_sources.add((direct.from_x, direct.from_y))
                continue

            staging = self._stage_toward_general(target_x, target_y, relaxed, reserved_sources)
            if staging is not None:
                moves.append(staging)
                reserved_sources.add((staging.from_x, staging.from_y))

        return moves[:3]

    def _visible_enemy_generals(self) -> list[tuple[int, int, Tile]]:
        targets: list[tuple[int, int, Tile]] = []
        for y, row in enumerate(self.board.tiles):
            for x, tile in enumerate(row):
                if (
                    tile.type == TileType.GENERAL
                    and tile.occupier is not None
                    and tile.occupier >= 0
                    and tile.occupier != self.player_idx
                ):
                    targets.append((x, y, tile))
        targets.sort(key=lambda item: (item[2].army, self._nearest_owned_distance(item[0], item[1]), item[0], item[1]))
        return targets

    def _direct_general_attack(
        self,
        target_x: int,
        target_y: int,
        target_tile: Tile,
        constraints: dict,
        used_sources: set[tuple[int, int]],
    ) -> Move | None:
        candidates: list[tuple[int, int, int]] = []
        for nx, ny in self._neighbors(target_x, target_y):
            if (nx, ny) in used_sources:
                continue
            source = self.board.tiles[ny][nx]
            if source.occupier != self.player_idx or source.army <= 1:
                continue
            full_moving = source.army - 1
            half_moving = source.army // 2
            can_kill_full = full_moving > target_tile.army
            can_kill_half = half_moving > target_tile.army
            if not can_kill_full and not can_kill_half:
                continue
            candidates.append((-max(full_moving, half_moving), nx, ny))

        for _, source_x, source_y in sorted(candidates):
            source = self.board.tiles[source_y][source_x]
            preferred = [False, True] if source.army - 1 > target_tile.army else [True, False]
            for take_half in preferred:
                moving = self._moving_army(source.army, take_half)
                if moving <= target_tile.army:
                    continue
                move = self._build_constrained_move(source_x, source_y, target_x, target_y, constraints, take_half)
                if move is not None:
                    return move
        return None

    def _stage_toward_general(
        self,
        target_x: int,
        target_y: int,
        constraints: dict,
        used_sources: set[tuple[int, int]],
    ) -> Move | None:
        target_neighbors = {
            (nx, ny)
            for nx, ny in self._neighbors(target_x, target_y)
            if self.board.tiles[ny][nx].type != TileType.MOUNTAIN
            and self.board.tiles[ny][nx].occupier != TILE_FOG_OBSTACLE
        }
        candidates: list[tuple[int, int, int, int]] = []
        for x, y, tile in self._own_tiles():
            if (x, y) in used_sources or tile.army <= 1:
                continue
            distance = self._manhattan(x, y, target_x, target_y)
            if distance > 4:
                continue
            candidates.append((-tile.army, distance, x, y))

        for _, _, x, y in sorted(candidates):
            step = self._best_decapitation_step(x, y, target_x, target_y, target_neighbors, constraints)
            if step is None:
                continue
            nx, ny = step
            move = self._build_constrained_move(x, y, nx, ny, constraints)
            if move is not None:
                return move
        return None

    def _best_decapitation_step(
        self,
        from_x: int,
        from_y: int,
        target_x: int,
        target_y: int,
        target_neighbors: set[tuple[int, int]],
        constraints: dict,
    ) -> tuple[int, int] | None:
        current_distance = self._manhattan(from_x, from_y, target_x, target_y)
        candidates: list[tuple[int, int, int, int]] = []
        for nx, ny in self._neighbors(from_x, from_y):
            if (nx, ny) == (target_x, target_y):
                continue
            if not self._destination_allowed(nx, ny, constraints):
                continue
            tile = self.board.tiles[ny][nx]
            if tile.occupier not in (self.player_idx, None, TILE_FOG):
                continue
            distance = self._manhattan(nx, ny, target_x, target_y)
            if distance >= current_distance:
                continue
            staging_rank = 0 if (nx, ny) in target_neighbors else 1
            owned_rank = 0 if tile.occupier == self.player_idx else 1
            candidates.append((staging_rank, distance, owned_rank, self._pack(nx, ny)))
        if not candidates:
            return None
        _, _, _, packed = min(candidates)
        return self._unpack(packed)

    def _safe_expansion_fallback(self, used_sources: set[tuple[int, int]]) -> list[Move]:
        relaxed = {
            **self.base_constraints,
            "min_general_garrison": 1,
            "min_city_garrison": 1,
            "max_commitment_percent": 100,
            "avoid_fog": False,
        }
        candidates: list[tuple[int, int, int, int, int]] = []
        for x, y, tile in self._own_tiles():
            if (x, y) in used_sources or tile.army <= 1:
                continue
            for nx, ny in self._neighbors(x, y):
                target = self.board.tiles[ny][nx]
                if target.type == TileType.MOUNTAIN or target.occupier == TILE_FOG_OBSTACLE:
                    continue
                if target.occupier == self.player_idx:
                    continue
                if target.occupier is None:
                    rank = 0
                elif target.occupier == TILE_FOG:
                    rank = 1
                elif target.occupier is not None and target.occupier >= 0:
                    rank = 2
                else:
                    rank = 3
                candidates.append((rank, -tile.army, x, y, self._pack(nx, ny)))

        moves: list[Move] = []
        for _, _, x, y, packed_dest in sorted(candidates):
            if (x, y) in used_sources:
                continue
            nx, ny = self._unpack(packed_dest)
            move = self._build_constrained_move(x, y, nx, ny, relaxed, take_half=True)
            if move is None:
                move = self._build_constrained_move(x, y, nx, ny, relaxed)
            if move is not None:
                moves.append(move)
                used_sources.add((x, y))
                break
        return moves

    def _endgame_hunt_moves(
        self,
        strategy: dict,
        constraints: dict,
        used_sources: set[tuple[int, int]],
    ) -> list[Move]:
        if not self._is_endgame_superior(strategy):
            return []

        relaxed = {
            **constraints,
            "min_general_garrison": 1,
            "min_city_garrison": 1,
            "max_commitment_percent": 100,
            "avoid_fog": False,
        }
        sources = [
            (x, y, tile)
            for x, y, tile in self._own_tiles()
            if (x, y) not in used_sources and tile.army > 1
        ]
        if not sources:
            return []

        enemy_targets = self._visible_enemy_targets()
        fog_targets = self._frontier_targets({TILE_FOG})
        neutral_targets = self._frontier_targets({None})
        targets = enemy_targets or fog_targets or neutral_targets
        if not targets:
            return []

        sources.sort(key=lambda item: (-item[2].army, self._frontier_distance(item[0], item[1]), item[1], item[0]))
        moves: list[Move] = []
        reserved_sources = set(used_sources)
        reserved_destinations: set[tuple[int, int]] = set()

        for x, y, tile in sources:
            if len(moves) >= 8:
                break
            if (x, y) in reserved_sources:
                continue
            step = self._hunt_step(x, y, targets, relaxed, reserved_destinations)
            if step is None:
                continue
            nx, ny = step
            take_half = tile.type in (TileType.GENERAL, TileType.CITY)
            move = self._build_constrained_move(x, y, nx, ny, relaxed, take_half=take_half)
            if move is None:
                move = self._build_constrained_move(x, y, nx, ny, relaxed)
            if move is None:
                continue
            moves.append(move)
            reserved_sources.add((x, y))
            reserved_destinations.add((nx, ny))

        return moves

    def _is_endgame_superior(self, strategy: dict) -> bool:
        context = strategy.get("_context") or {}
        stats = context.get("stats") or {}
        rankings = context.get("rankings") or []
        if not rankings:
            return False

        alive_enemies = [
            ranking
            for ranking in rankings
            if ranking.get("player") != self.player_idx and ranking.get("alive")
        ]
        if not alive_enemies:
            return False

        own_army = int(stats.get("army", 0))
        max_enemy_army = max(1, max(int(enemy.get("army", 0)) for enemy in alive_enemies))
        total_enemy_army = max(1, sum(int(enemy.get("army", 0)) for enemy in alive_enemies))
        own_tiles = int(stats.get("tiles", 0))
        max_enemy_tiles = max(1, max(int(enemy.get("tiles", 0)) for enemy in alive_enemies))
        return (
            own_army >= 120
            and own_tiles >= max_enemy_tiles
            and (
                own_army >= max_enemy_army * 2
                or own_army >= total_enemy_army * 1.05
                or own_army - max_enemy_army >= 250
            )
        )

    def _visible_enemy_targets(self) -> list[tuple[int, int, int]]:
        targets: list[tuple[int, int, int]] = []
        for y, row in enumerate(self.board.tiles):
            for x, tile in enumerate(row):
                if tile.occupier is None or tile.occupier < 0 or tile.occupier == self.player_idx:
                    continue
                if tile.type == TileType.GENERAL:
                    rank = 0
                elif tile.type == TileType.CITY:
                    rank = 1
                else:
                    rank = 2
                targets.append((rank, x, y))
        targets.sort(key=lambda item: (item[0], self._nearest_owned_distance(item[1], item[2]), item[2], item[1]))
        return targets

    def _frontier_targets(self, occupiers: set[int | None]) -> list[tuple[int, int, int]]:
        targets: list[tuple[int, int, int]] = []
        for y, row in enumerate(self.board.tiles):
            for x, tile in enumerate(row):
                if tile.type == TileType.MOUNTAIN or tile.occupier == TILE_FOG_OBSTACLE:
                    continue
                if tile.occupier not in occupiers:
                    continue
                adjacent_own = any(
                    self.board.tiles[ny][nx].occupier == self.player_idx
                    for nx, ny in self._neighbors(x, y)
                )
                if not adjacent_own:
                    continue
                if tile.occupier == TILE_FOG:
                    rank = 3
                elif tile.type == TileType.CITY:
                    rank = 4
                else:
                    rank = 5
                targets.append((rank, x, y))
        targets.sort(key=lambda item: (item[0], self._nearest_owned_distance(item[1], item[2]), item[2], item[1]))
        return targets

    def _hunt_step(
        self,
        from_x: int,
        from_y: int,
        targets: list[tuple[int, int, int]],
        constraints: dict,
        reserved_destinations: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        path_candidates: list[tuple[int, int, int, int, list[tuple[int, int]]]] = []
        for rank, target_x, target_y in targets[:12]:
            path = self._path_to(from_x, from_y, target_x, target_y, constraints, prefer_owned=False)
            if path is None or len(path) < 2:
                continue
            nx, ny = path[1]
            if (nx, ny) in reserved_destinations:
                continue
            path_candidates.append((rank, len(path), target_y, target_x, path))
        if path_candidates:
            _, _, _, _, path = min(path_candidates)
            return path[1]

        best_target = min(
            targets,
            key=lambda item: (
                self._manhattan(from_x, from_y, item[1], item[2]),
                item[0],
                item[2],
                item[1],
            ),
        )
        _, target_x, target_y = best_target

        current_distance = self._manhattan(from_x, from_y, target_x, target_y)
        candidates: list[tuple[int, int, int, int, int]] = []
        for nx, ny in self._neighbors(from_x, from_y):
            if (nx, ny) in reserved_destinations:
                continue
            if not self._destination_allowed(nx, ny, constraints):
                continue
            tile = self.board.tiles[ny][nx]
            if tile.occupier == self.player_idx:
                destination_rank = 3
            elif tile.occupier is not None and tile.occupier >= 0:
                destination_rank = 0
            elif tile.occupier == TILE_FOG:
                destination_rank = 1
            elif tile.occupier is None:
                destination_rank = 2
            else:
                destination_rank = 4
            distance = self._manhattan(nx, ny, target_x, target_y)
            if distance > current_distance and destination_rank >= 3:
                continue
            candidates.append((destination_rank, distance, tile.army, ny, nx))
        if not candidates:
            return None
        _, _, _, ny, nx = min(candidates)
        return nx, ny

    def _frontier_distance(self, x: int, y: int) -> int:
        best = 9999
        for ty, row in enumerate(self.board.tiles):
            for tx, tile in enumerate(row):
                if tile.type == TileType.MOUNTAIN or tile.occupier == TILE_FOG_OBSTACLE:
                    continue
                if tile.occupier == self.player_idx:
                    continue
                adjacent_own = any(
                    self.board.tiles[ny][nx].occupier == self.player_idx
                    for nx, ny in self._neighbors(tx, ty)
                )
                if adjacent_own:
                    best = min(best, self._manhattan(x, y, tx, ty))
        return best

    def _blocked_reason(
        self,
        objective: dict,
        constraints: dict,
        used_sources: set[tuple[int, int]],
    ) -> str:
        objective_type = objective.get("type")
        movable_sources = [
            (x, y, tile)
            for x, y, tile in self._own_tiles()
            if (x, y) not in used_sources and tile.army > 1
        ]
        if not movable_sources:
            return "no_surplus_source"
        if objective_type in {"expand_region", "reinforce_region", "defend_region"}:
            region = self._parse_region(objective.get("region"))
            if region is None:
                return "invalid_region"
        if objective_type == "attack_position":
            target = objective.get("target") or {}
            if not self._is_int(target.get("x")) or not self._is_int(target.get("y")):
                return "invalid_target"
        if constraints.get("avoid_fog"):
            return "constraints_or_avoid_fog"
        return "constraints_or_no_target"

    def _best_step_toward(
        self,
        from_x: int,
        from_y: int,
        target_x: int,
        target_y: int,
        constraints: dict,
        prefer_owned: bool,
    ) -> tuple[int, int] | None:
        path_step = self._best_path_step(from_x, from_y, target_x, target_y, constraints, prefer_owned)
        if path_step is not None:
            return path_step

        current_distance = self._manhattan(from_x, from_y, target_x, target_y)
        candidates: list[tuple[int, int, int, int]] = []
        for nx, ny in self._neighbors(from_x, from_y):
            if not self._destination_allowed(nx, ny, constraints):
                continue
            distance = self._manhattan(nx, ny, target_x, target_y)
            if distance > current_distance:
                continue
            tile = self.board.tiles[ny][nx]
            owned_rank = 0 if tile.occupier == self.player_idx else 1
            if not prefer_owned:
                owned_rank = 0
            candidates.append((distance, owned_rank, nx, ny))
        if not candidates:
            return None
        _, _, nx, ny = min(candidates)
        return nx, ny

    def _best_path_step(
        self,
        from_x: int,
        from_y: int,
        target_x: int,
        target_y: int,
        constraints: dict,
        prefer_owned: bool,
    ) -> tuple[int, int] | None:
        path = self._path_to(from_x, from_y, target_x, target_y, constraints, prefer_owned)
        if path is None or len(path) < 2:
            return None
        return path[1]

    def _path_to(
        self,
        from_x: int,
        from_y: int,
        target_x: int,
        target_y: int,
        constraints: dict,
        prefer_owned: bool,
    ) -> list[tuple[int, int]] | None:
        if (from_x, from_y) == (target_x, target_y):
            return [(from_x, from_y)]
        if not self.board.in_bounds(target_x, target_y):
            return None

        queue = deque([(from_x, from_y)])
        previous: dict[tuple[int, int], tuple[int, int] | None] = {(from_x, from_y): None}

        while queue:
            x, y = queue.popleft()
            if (x, y) == (target_x, target_y):
                break
            neighbors = sorted(
                self._neighbors(x, y),
                key=lambda item: self._path_tile_rank(item[0], item[1], prefer_owned),
            )
            for nx, ny in neighbors:
                if (nx, ny) in previous:
                    continue
                if not self._destination_allowed(nx, ny, constraints):
                    continue
                tile = self.board.tiles[ny][nx]
                if (nx, ny) != (target_x, target_y) and not self._is_path_passable(tile):
                    continue
                previous[(nx, ny)] = (x, y)
                queue.append((nx, ny))

        if (target_x, target_y) not in previous:
            return None

        path = [(target_x, target_y)]
        while previous[path[-1]] is not None:
            parent = previous[path[-1]]
            if parent is None:
                return None
            path.append(parent)
        path.reverse()
        return path

    def _is_path_passable(self, tile: Tile) -> bool:
        if tile.type == TileType.MOUNTAIN or tile.occupier == TILE_FOG_OBSTACLE:
            return False
        return tile.occupier in (self.player_idx, None, TILE_FOG) or (
            tile.occupier is not None and tile.occupier >= 0
        )

    def _path_tile_rank(self, x: int, y: int, prefer_owned: bool) -> tuple[int, int, int]:
        tile = self.board.tiles[y][x]
        if tile.occupier is not None and tile.occupier >= 0 and tile.occupier != self.player_idx:
            rank = 0
        elif tile.occupier == TILE_FOG:
            rank = 1
        elif tile.occupier is None:
            rank = 2
        elif tile.occupier == self.player_idx:
            rank = 0 if prefer_owned else 3
        else:
            rank = 9
        return rank, y, x

    def _weak_tiles_in_region(self, region: Region) -> list[tuple[int, int, Tile]]:
        own_tiles = list(self._own_tiles(region))
        if not own_tiles:
            return []
        average_army = sum(tile.army for _, _, tile in own_tiles) / len(own_tiles)
        threshold = max(3, int(average_army * 0.5))
        return [(x, y, tile) for x, y, tile in own_tiles if tile.army <= threshold]

    def _is_border_tile(self, x: int, y: int, region: Region) -> bool:
        if x in (region.x1, region.x2) or y in (region.y1, region.y2):
            return True
        for nx, ny in self._neighbors(x, y):
            if not region.contains(nx, ny):
                return True
            neighbor = self.board.tiles[ny][nx]
            if neighbor.occupier not in (self.player_idx, TILE_FOG_OBSTACLE) and neighbor.type != TileType.MOUNTAIN:
                return True
        return False

    def _is_expand_target(self, tile: Tile) -> bool:
        return tile.occupier is None or tile.occupier == TILE_FOG

    def _own_tiles(self, region: Region | None = None) -> list[tuple[int, int, Tile]]:
        tiles: list[tuple[int, int, Tile]] = []
        for y, row in enumerate(self.board.tiles):
            for x, tile in enumerate(row):
                if tile.occupier != self.player_idx:
                    continue
                if region is not None and not region.contains(x, y):
                    continue
                tiles.append((x, y, tile))
        return tiles

    def _parse_region(self, raw_region: Any) -> Region | None:
        if not isinstance(raw_region, dict):
            return None
        if not all(self._is_int(raw_region.get(key)) for key in ("x1", "y1", "x2", "y2")):
            return None
        x1 = max(0, min(int(raw_region["x1"]), int(raw_region["x2"])))
        x2 = min(self.board.width - 1, max(int(raw_region["x1"]), int(raw_region["x2"])))
        y1 = max(0, min(int(raw_region["y1"]), int(raw_region["y2"])))
        y2 = min(self.board.height - 1, max(int(raw_region["y1"]), int(raw_region["y2"])))
        if x1 > x2 or y1 > y2:
            return None
        return Region(x1, y1, x2, y2)

    def _neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        candidates = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        return [(nx, ny) for nx, ny in candidates if self.board.in_bounds(nx, ny)]

    def _total_visible_army(self) -> int:
        return sum(tile.army for _, _, tile in self._own_tiles())

    def _nearest_owned_distance(self, x: int, y: int) -> int:
        owned = self._own_tiles()
        if not owned:
            return 9999
        return min(self._manhattan(x, y, ox, oy) for ox, oy, _ in owned)

    @staticmethod
    def _moving_army(source_army: int, take_half: bool) -> int:
        return source_army // 2 if take_half else source_army - 1

    @staticmethod
    def _nearest_distance(x: int, y: int, targets: list[tuple[int, int, Tile]]) -> int:
        return min(abs(x - tx) + abs(y - ty) for tx, ty, _ in targets)

    @staticmethod
    def _manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
        return abs(x1 - x2) + abs(y1 - y2)

    @staticmethod
    def _distance_to_point(x: int, y: int, point: tuple[float, float]) -> float:
        return abs(x - point[0]) + abs(y - point[1])

    @staticmethod
    def _pack(x: int, y: int) -> int:
        return (y << 16) + x

    @staticmethod
    def _unpack(value: int) -> tuple[int, int]:
        return value & 0xFFFF, value >> 16

    @staticmethod
    def _is_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)
