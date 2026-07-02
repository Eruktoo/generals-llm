from __future__ import annotations

import random

from .board import TILE_FOG, TILE_FOG_OBSTACLE, Board
from .types import Move, TileType


class RandomBot:
    def __init__(self, player_idx: int, seed: int | None = None):
        self.player_idx = player_idx
        self.rng = random.Random(seed)

    def request_move(self, board_view: dict) -> list[Move]:
        board: Board = board_view["board"]
        moves: list[Move] = []

        for y, row in enumerate(board.tiles):
            for x, tile in enumerate(row):
                if tile.occupier != self.player_idx or tile.army <= 1:
                    continue

                candidates = []
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if not board.in_bounds(nx, ny):
                        continue
                    target = board.tiles[ny][nx]
                    if target.type == TileType.MOUNTAIN:
                        continue
                    if target.occupier == TILE_FOG_OBSTACLE:
                        continue
                    candidates.append((nx, ny))

                if candidates:
                    to_x, to_y = self.rng.choice(candidates)
                    moves.append(Move(x, y, to_x, to_y, self.rng.choice([False, True])))

        return moves

