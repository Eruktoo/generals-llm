from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TileType(Enum):
    PLAIN = auto()
    MOUNTAIN = auto()
    CITY = auto()
    GENERAL = auto()
    SWAMP = auto()
    DESERT = auto()


@dataclass
class Tile:
    type: TileType
    occupier: int | None = None
    army: int = 0


@dataclass(frozen=True)
class Move:
    from_x: int
    from_y: int
    to_x: int
    to_y: int
    take_half: bool = False


@dataclass
class PlayerState:
    index: int
    alive: bool = True
    name: str = ""

