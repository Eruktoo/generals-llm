from __future__ import annotations

from collections import defaultdict

from .board import Board
from .types import Move, PlayerState, TileType


class Game:
    def __init__(self, board: Board | None = None, players: list[PlayerState] | None = None):
        self.board = board
        self.players = players or []
        self.turn = 0
        self.phase = 0
        self.alive: list[bool] = []
        self.generals: dict[int, tuple[int, int]] = {}
        self.cities: list[tuple[int, int]] = []
        self.kills: defaultdict[int, int] = defaultdict(int)

        if board is not None and players:
            self.init()

    def init(
        self,
        width: int | None = None,
        height: int | None = None,
        num_players: int | None = None,
        seed: int | None = None,
    ) -> None:
        if self.board is None:
            if width is None or height is None or num_players is None:
                raise ValueError("width, height, and num_players are required without a board")
            self.board = Board.generate_map(width, height, num_players, seed)
            self.players = [PlayerState(i, True, f"Player {i}") for i in range(num_players)]

        if not self.players:
            player_indexes = sorted({
                tile.occupier
                for row in self.board.tiles
                for tile in row
                if tile.occupier is not None and tile.occupier >= 0
            })
            self.players = [PlayerState(i, True, f"Player {i}") for i in player_indexes]

        self.turn = 0
        self.phase = 0
        self.alive = [player.alive for player in self.players]
        self.generals = {}
        self.cities = []
        self.kills = defaultdict(int)

        for y, row in enumerate(self.board.tiles):
            for x, tile in enumerate(row):
                if tile.type == TileType.GENERAL and tile.occupier is not None:
                    self.generals[tile.occupier] = (x, y)
                elif tile.type == TileType.CITY:
                    self.cities.append((x, y))

    def step(self, moves: dict[int, list[Move]] | None = None) -> None:
        self.execute_moves(moves or {})
        if self.phase == 0:
            self._produce_armies()
            self.phase = 1
        else:
            self.phase = 0
            self.turn += 1

    def execute_moves(self, moves_dict: dict[int, list[Move]]) -> None:
        # Flatten all moves with priority info for proper ordering
        flat_moves: list[tuple[int, int, Move]] = []  # (priority, army_size, move)
        for player_idx in sorted(moves_dict):
            if not self._is_alive(player_idx):
                continue
            for move in moves_dict[player_idx]:
                if not self._valid_move(player_idx, move):
                    continue
                source = self.board.tiles[move.from_y][move.from_x]
                army_size = source.army // 2 if move.take_half else source.army - 1
                # Priority: chase(0) < defensive(1) < normal(2) < attack_general(3)
                priority = self._move_priority(player_idx, move)
                flat_moves.append((priority, -army_size, player_idx, move))

        # Sort: higher priority first, then larger army first
        flat_moves.sort(key=lambda x: (-x[0], x[1], x[2]))

        for _, _, player_idx, move in flat_moves:
            if not self._is_alive(player_idx):
                continue
            self._execute_move(player_idx, move)

    def _move_priority(self, player_idx: int, move: Move) -> int:
        """Higher number = higher priority."""
        target = self.board.tiles[move.to_y][move.to_x]
        # Attack on enemy general: lowest priority (0)
        if target.type == TileType.GENERAL and target.occupier != player_idx and target.occupier is not None:
            return 0
        # Normal attack: medium priority (1)
        if target.occupier != player_idx and target.occupier is not None and target.occupier >= 0:
            return 1
        # Defensive (friendly merge): high priority (2)
        if target.occupier == player_idx:
            return 2
        # Expand to neutral: also high priority (2)
        if target.occupier is None or target.occupier < 0:
            return 2
        return 1

    def get_player_view(self, player_idx: int) -> dict:
        visible_board = self.board.get_visible(player_idx)
        stats = self._player_stats(player_idx)
        return {
            "player": player_idx,
            "turn": self.turn,
            "phase": self.phase,
            "board": visible_board,
            "stats": stats,
            "rankings": self.get_rankings(),
        }

    def get_rankings(self) -> list[dict]:
        rankings = []
        for player in self.players:
            stats = self._player_stats(player.index)
            rankings.append({
                "player": player.index,
                "army": stats["army"],
                "tiles": stats["tiles"],
                "alive": self._is_alive(player.index),
                "kills": self.kills[player.index],
            })
        rankings.sort(key=lambda item: (item["alive"], item["tiles"], item["army"]), reverse=True)
        return rankings

    def is_finished(self) -> bool:
        return sum(1 for alive in self.alive if alive) <= 1

    def _execute_move(self, player_idx: int, move: Move) -> None:
        if not self._valid_move(player_idx, move):
            return

        source = self.board.tiles[move.from_y][move.from_x]
        target = self.board.tiles[move.to_y][move.to_x]
        moving_army = source.army // 2 if move.take_half else source.army - 1
        if moving_army <= 0:
            return

        source.army -= moving_army

        if target.occupier == player_idx:
            target.army += moving_army
            return

        remaining = moving_army - target.army
        if remaining > 0:
            defeated_player = target.occupier if target.occupier is not None and target.occupier >= 0 else None
            was_general = target.type == TileType.GENERAL and defeated_player is not None
            target.occupier = player_idx
            target.army = remaining

            if was_general:
                self._capture_player(attacker=player_idx, defeated=defeated_player)
        else:
            target.army = -remaining

    def _valid_move(self, player_idx: int, move: Move) -> bool:
        if self.board is None:
            return False
        if not self.board.in_bounds(move.from_x, move.from_y):
            return False
        if not self.board.in_bounds(move.to_x, move.to_y):
            return False
        if abs(move.from_x - move.to_x) + abs(move.from_y - move.to_y) != 1:
            return False

        source = self.board.tiles[move.from_y][move.from_x]
        target = self.board.tiles[move.to_y][move.to_x]
        return (
            source.occupier == player_idx
            and source.army > 1
            and target.type != TileType.MOUNTAIN
        )

    def _capture_player(self, attacker: int, defeated: int) -> None:
        if defeated == attacker or not self._is_alive(defeated):
            return

        self.kills[attacker] += 1
        self.alive[defeated] = False
        if 0 <= defeated < len(self.players):
            self.players[defeated].alive = False

        for row in self.board.tiles:
            for tile in row:
                if tile.occupier == defeated:
                    tile.occupier = attacker
                    tile.army //= 2

        self.generals.pop(defeated, None)

    def _produce_armies(self) -> None:
        for row in self.board.tiles:
            for tile in row:
                if tile.occupier is None or not self._is_alive(tile.occupier):
                    continue
                if tile.type in (TileType.CITY, TileType.GENERAL):
                    tile.army += 1
                elif self.turn > 0 and self.turn % 25 == 0:
                    tile.army += 1

    def _player_stats(self, player_idx: int) -> dict:
        army = 0
        tiles = 0
        cities = 0
        for row in self.board.tiles:
            for tile in row:
                if tile.occupier == player_idx:
                    tiles += 1
                    army += tile.army
                    if tile.type == TileType.CITY:
                        cities += 1
        return {
            "army": army,
            "tiles": tiles,
            "cities": cities,
            "alive": self._is_alive(player_idx),
            "general": self.generals.get(player_idx),
        }

    def _is_alive(self, player_idx: int) -> bool:
        return 0 <= player_idx < len(self.alive) and self.alive[player_idx]

