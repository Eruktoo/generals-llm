from __future__ import annotations

import random
from collections import deque
from copy import deepcopy
from typing import TypeAlias

from .types import Tile, TileType


TILE_FOG = -3
TILE_FOG_OBSTACLE = -4

BoardView: TypeAlias = "Board"


class Board:
    def __init__(self, width: int, height: int, tiles: list[list[Tile]] | None = None):
        if width <= 0 or height <= 0:
            raise ValueError("Board dimensions must be positive")
        self.width = width
        self.height = height
        self.tiles = tiles if tiles is not None else [
            [Tile(TileType.PLAIN) for _ in range(width)] for _ in range(height)
        ]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def adjacent4(self, x: int, y: int) -> list[tuple[int, int]]:
        candidates = ((x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        return [(cx, cy) for cx, cy in candidates if self.in_bounds(cx, cy)]

    def copy(self) -> "Board":
        return Board(self.width, self.height, deepcopy(self.tiles))

    def get_visible(self, player_idx: int) -> BoardView:
        visible: set[tuple[int, int]] = set()

        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if tile.occupier == player_idx or tile.type == TileType.CITY:
                    visible.update(self.adjacent4(x, y))

        view_tiles: list[list[Tile]] = []
        for y, row in enumerate(self.tiles):
            view_row: list[Tile] = []
            for x, tile in enumerate(row):
                if (x, y) in visible:
                    view_row.append(deepcopy(tile))
                elif tile.type in (TileType.MOUNTAIN, TileType.CITY):
                    view_row.append(Tile(tile.type, TILE_FOG_OBSTACLE, 0))
                else:
                    view_row.append(Tile(TileType.PLAIN, TILE_FOG, 0))
            view_tiles.append(view_row)

        return Board(self.width, self.height, view_tiles)

    @staticmethod
    def generate_map(width: int, height: int, num_players: int, seed: int | None = None) -> "Board":
        if num_players <= 0:
            raise ValueError("num_players must be positive")
        if width * height < num_players:
            raise ValueError("Board is too small for the requested players")

        rng = random.Random(seed)
        board = Board(width, height)
        start_positions = Board._starting_positions(width, height, num_players)
        protected = Board._protected_start_positions(width, height, start_positions)

        mountain_count = int(width * height * 0.15)
        all_positions = [
            (x, y)
            for y in range(height)
            for x in range(width)
            if (x, y) not in protected
        ]
        for x, y in rng.sample(all_positions, min(mountain_count, len(all_positions))):
            board.tiles[y][x].type = TileType.MOUNTAIN

        free_positions = [
            (x, y)
            for y in range(height)
            for x in range(width)
            if board.tiles[y][x].type != TileType.MOUNTAIN and (x, y) not in protected
        ]
        city_count = min(
            rng.randint(max(6, int(width * height * 0.06)), max(12, int(width * height * 0.09))),
            max(0, len(free_positions) - num_players),
        )
        for x, y in rng.sample(free_positions, city_count):
            board.tiles[y][x] = Tile(TileType.CITY, None, rng.randint(35, 50))

        Board._connect_open_terrain(board, start_positions)
        for player_idx, (x, y) in enumerate(start_positions):
            board.tiles[y][x] = Tile(TileType.GENERAL, player_idx, 0)

        return board

    @staticmethod
    def _protected_start_positions(
        width: int,
        height: int,
        starts: list[tuple[int, int]],
        radius: int = 2,
    ) -> set[tuple[int, int]]:
        protected: set[tuple[int, int]] = set()
        for sx, sy in starts:
            for dx in range(-radius, radius + 1):
                remaining = radius - abs(dx)
                for dy in range(-remaining, remaining + 1):
                    x = sx + dx
                    y = sy + dy
                    if 0 <= x < width and 0 <= y < height:
                        protected.add((x, y))
        return protected

    @staticmethod
    def _connect_open_terrain(board: "Board", starts: list[tuple[int, int]]) -> None:
        passable = Board._passable_positions(board)
        if not passable:
            return

        main_component = Board._component_from(board, starts[0] if starts else next(iter(passable)))
        for start in starts[1:]:
            if start not in main_component:
                Board._carve_path_to_component(board, start, main_component)
                main_component = Board._component_from(board, starts[0])

        while True:
            passable = Board._passable_positions(board)
            remaining = passable - main_component
            if not remaining:
                return
            component_start = min(remaining, key=lambda item: (item[1], item[0]))
            Board._carve_path_to_component(board, component_start, main_component)
            main_component = Board._component_from(board, starts[0] if starts else component_start)

    @staticmethod
    def _passable_positions(board: "Board") -> set[tuple[int, int]]:
        return {
            (x, y)
            for y, row in enumerate(board.tiles)
            for x, tile in enumerate(row)
            if tile.type not in (TileType.MOUNTAIN, TileType.CITY)
        }

    @staticmethod
    def _component_from(board: "Board", start: tuple[int, int]) -> set[tuple[int, int]]:
        sx, sy = start
        if not board.in_bounds(sx, sy):
            return set()
        if board.tiles[sy][sx].type in (TileType.MOUNTAIN, TileType.CITY):
            board.tiles[sy][sx] = Tile(TileType.PLAIN)

        component = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not board.in_bounds(nx, ny) or (nx, ny) in component:
                    continue
                if board.tiles[ny][nx].type in (TileType.MOUNTAIN, TileType.CITY):
                    continue
                component.add((nx, ny))
                queue.append((nx, ny))
        return component

    @staticmethod
    def _carve_path_to_component(
        board: "Board",
        start: tuple[int, int],
        target_component: set[tuple[int, int]],
    ) -> None:
        queue = deque([start])
        previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        target: tuple[int, int] | None = None

        while queue:
            x, y = queue.popleft()
            if (x, y) in target_component:
                target = (x, y)
                break
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not board.in_bounds(nx, ny) or (nx, ny) in previous:
                    continue
                previous[(nx, ny)] = (x, y)
                queue.append((nx, ny))

        if target is None:
            return

        cursor: tuple[int, int] | None = target
        while cursor is not None:
            x, y = cursor
            if board.tiles[y][x].type in (TileType.MOUNTAIN, TileType.CITY):
                board.tiles[y][x] = Tile(TileType.PLAIN)
            cursor = previous[cursor]

    @staticmethod
    def _starting_positions(width: int, height: int, num_players: int) -> list[tuple[int, int]]:
        if num_players == 1:
            return [(width // 2, height // 2)]

        min_x = 1 if width > 2 else 0
        max_x = width - 2 if width > 2 else width - 1
        min_y = 1 if height > 2 else 0
        max_y = height - 2 if height > 2 else height - 1

        anchors = [
            (min_x, min_y),
            (max_x, max_y),
            (max_x, min_y),
            (min_x, max_y),
            (width // 2, min_y),
            (width // 2, max_y),
            (min_x, height // 2),
            (max_x, height // 2),
        ]
        if num_players <= len(anchors):
            return anchors[:num_players]

        positions = anchors[:]
        for idx in range(len(anchors), num_players):
            x = round((idx + 1) * (width - 1) / (num_players + 1))
            y = round(((idx * 3) % num_players + 1) * (height - 1) / (num_players + 1))
            candidate = (x, y)
            while candidate in positions:
                x = (x + 1) % width
                if x == 0:
                    y = (y + 1) % height
                candidate = (x, y)
            positions.append(candidate)
        return positions[:num_players]
