from __future__ import annotations

import os
import sys
import time

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.board import TILE_FOG, TILE_FOG_OBSTACLE, Board
from engine.game import Game
from engine.random_bot import RandomBot
from engine.types import PlayerState, TileType


def tile_char(tile) -> str:
    if tile.occupier == TILE_FOG:
        return "#"
    if tile.occupier == TILE_FOG_OBSTACLE:
        return "?"
    if tile.occupier is not None:
        return str(tile.occupier % 10)
    if tile.type == TileType.MOUNTAIN:
        return "M"
    if tile.type == TileType.CITY:
        return "C"
    if tile.type == TileType.GENERAL:
        return "G"
    return "."


def army_text(tile) -> str:
    if tile.occupier in (TILE_FOG, TILE_FOG_OBSTACLE):
        return "?"
    return str(tile.army)


def print_board(board: Board) -> None:
    for row in board.tiles:
        chars = [f"{tile_char(tile)}:{army_text(tile):>3}" for tile in row]
        print(" ".join(chars))


def print_rankings(game: Game) -> None:
    print("Rankings:")
    for rank, row in enumerate(game.get_rankings(), start=1):
        status = "alive" if row["alive"] else "dead"
        print(
            f"  {rank}. P{row['player']} {status} "
            f"army={row['army']} tiles={row['tiles']} kills={row['kills']}"
        )


def main() -> None:
    board = Board.generate_map(12, 12, 4, seed=42)
    players = [PlayerState(i, True, f"Bot {i}") for i in range(4)]
    game = Game(board, players)
    bots = [RandomBot(i, seed=100 + i) for i in range(4)]

    while not game.is_finished() and game.turn < 500:
        print(f"\nTurn {game.turn} phase {game.phase}")
        print_board(game.board)
        moves = {
            bot.player_idx: bot.request_move(game.get_player_view(bot.player_idx))
            for bot in bots
            if game.alive[bot.player_idx]
        }
        game.step(moves)
        if game.phase == 0:
            print_rankings(game)

    print(f"\nFinal turn {game.turn} phase {game.phase}")
    print_board(game.board)
    print_rankings(game)
    winners = [player.index for player in game.players if game.alive[player.index]]
    if winners:
        print(f"Winner: P{winners[0]}")
    else:
        print("Winner: none")


if __name__ == "__main__":
    main()
