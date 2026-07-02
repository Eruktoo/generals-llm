from __future__ import annotations

from dataclasses import dataclass
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
        constraints = {
            **self.base_constraints,
            **(strategy.get("constraints") or {}),
        }
        used_sources: set[tuple[int, int]] = set()
        moves: list[Move] = []

        for order in strategy.get("direct_orders") or []:
            move = self._move_from_direct_order(order, constraints, used_sources)
            if move is not None:
                moves.append(move)
                used_sources.add((move.from_x, move.from_y))

        objectives = sorted(
            strategy.get("objectives") or [],
            key=lambda objective: float(objective.get("priority", 0)),
            reverse=True,
        )
        for objective in objectives:
            for move in self._moves_for_objective(objective, constraints, used_sources):
                moves.append(move)
                used_sources.add((move.from_x, move.from_y))

        return moves

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
        max_stacks = 3 if commitment == "full" else 1
        sources = [
            (x, y, tile)
            for x, y, tile in self._own_tiles()
            if (x, y) not in used_sources and tile.army > 1
        ]
        sources.sort(key=lambda item: (-item[2].army, self._manhattan(item[0], item[1], target_x, target_y)))

        moves: list[Move] = []
        for x, y, _ in sources[:max_stacks]:
            step = self._best_step_toward(x, y, target_x, target_y, constraints, prefer_owned=False)
            if step is None:
                continue
            nx, ny = step
            move = self._build_constrained_move(x, y, nx, ny, constraints)
            if move is not None:
                moves.append(move)
        return moves

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

    def _best_step_toward(
        self,
        from_x: int,
        from_y: int,
        target_x: int,
        target_y: int,
        constraints: dict,
        prefer_owned: bool,
    ) -> tuple[int, int] | None:
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
