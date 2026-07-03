from __future__ import annotations

from typing import Any

from engine.board import TILE_FOG, TILE_FOG_OBSTACLE, Board
from engine.types import Tile, TileType


PERSONALITY_PROFILES = {
    "gambler": {
        "attack_bias": 1.25,
        "defense_bias": 0.55,
        "explore_bias": 0.95,
        "city_bias": 0.9,
        "direct_orders": 5,
        "dominance_ratio": 1.45,
        "dominance_stack_share": 0.55,
        "garrison": 5,
        "commitment": 95,
    },
    "conservative": {
        "attack_bias": 0.75,
        "defense_bias": 1.35,
        "explore_bias": 0.65,
        "city_bias": 1.2,
        "direct_orders": 2,
        "dominance_ratio": 2.4,
        "dominance_stack_share": 0.9,
        "garrison": 24,
        "commitment": 55,
    },
    "trickster": {
        "attack_bias": 0.95,
        "defense_bias": 0.9,
        "explore_bias": 1.05,
        "city_bias": 1.0,
        "direct_orders": 4,
        "dominance_ratio": 1.8,
        "dominance_stack_share": 0.7,
        "garrison": 12,
        "commitment": 75,
    },
    "crazy": {
        "attack_bias": 1.05,
        "defense_bias": 0.65,
        "explore_bias": 1.35,
        "city_bias": 0.75,
        "direct_orders": 6,
        "dominance_ratio": 1.6,
        "dominance_stack_share": 0.65,
        "garrison": 3,
        "commitment": 100,
    },
    "baseline": {
        "attack_bias": 1.0,
        "defense_bias": 1.0,
        "explore_bias": 1.0,
        "city_bias": 1.0,
        "direct_orders": 3,
        "dominance_ratio": 2.0,
        "dominance_stack_share": 0.7,
        "garrison": 10,
        "commitment": 75,
    },
}


class HeuristicAgent:
    """Deterministic baseline strategist that does not call an LLM."""

    def __init__(self, player_idx: int, personality: str = "baseline"):
        self.player_idx = player_idx
        self.personality = personality
        self.profile = PERSONALITY_PROFILES.get(personality, PERSONALITY_PROFILES["baseline"])

    def decide(self, game_view: dict) -> dict:
        board = self._require_board(game_view)
        stats = game_view.get("stats") or {}
        rankings = game_view.get("rankings") or []
        analysis = self._analyze_board(board)

        own_army = int(stats.get("army", 0))
        max_enemy_army = max(
            [
                int(row.get("army", 0))
                for row in rankings
                if row.get("player") != self.player_idx and row.get("alive")
            ]
            or [1]
        )
        alive_enemies = sum(
            1
            for row in rankings
            if row.get("player") != self.player_idx and row.get("alive")
        )
        max_stack = max((tile.army for _, _, tile in analysis["own_tiles"]), default=0)
        enemy_total = sum(
            int(row.get("army", 0))
            for row in rankings
            if row.get("player") != self.player_idx and row.get("alive")
        )
        dominant = (
            own_army >= max_enemy_army * float(self.profile["dominance_ratio"])
            or max_stack >= max(30, int(enemy_total * float(self.profile["dominance_stack_share"])))
        )
        contact = bool(analysis["enemy_tiles"])

        if (contact and self.profile["attack_bias"] >= 0.9) or dominant:
            stance = "aggressive"
        elif own_army < max_enemy_army * (0.65 * float(self.profile["defense_bias"])):
            stance = "defensive"
        else:
            stance = "balanced"

        constraints = self._constraints(stance, dominant)
        objectives: list[dict[str, Any]] = []

        attack_target = self._best_attack_target(analysis)
        if attack_target is not None:
            objectives.append({
                "type": "attack_position",
                "target": {"x": attack_target[0], "y": attack_target[1]},
                "priority": min(1.0, (1.0 if dominant else 0.85) * float(self.profile["attack_bias"])),
                "commitment": "full" if dominant or stance == "aggressive" else "limited",
            })

        pressure_target = attack_target or self._best_pressure_target(board, analysis, stats)
        if pressure_target is not None:
            objectives.append({
                "type": "attack_position",
                "target": {"x": pressure_target[0], "y": pressure_target[1]},
                "priority": min(1.0, (0.9 if dominant else 0.65) * float(self.profile["attack_bias"])),
                "commitment": "full" if dominant else "limited",
            })

        expand_region = self._expansion_region(board, stats, analysis, dominant)
        objectives.append({
            "type": "expand_region",
            "region": expand_region,
            "priority": min(1.0, (0.8 if not contact else 0.45) * float(self.profile["explore_bias"])),
        })

        if stance != "aggressive":
            objectives.append({
                "type": "defend_region",
                "region": self._defense_region(board, stats),
                "priority": min(1.0, 0.7 * float(self.profile["defense_bias"])),
            })
        objectives.append({
            "type": "reinforce_region",
            "region": self._frontier_region(board, stats, analysis),
            "priority": min(1.0, (0.55 if dominant else 0.65) * float(self.profile["defense_bias"])),
        })

        return {
            "stance": stance,
            "round_plan": self._plan_text(stance, dominant, alive_enemies, contact),
            "reasoning": (
                f"heuristic baseline: own_army={own_army}, max_enemy_army={max_enemy_army}, "
                f"max_stack={max_stack}, enemy_total={enemy_total}, contact={contact}, profile={self.personality}"
            ),
            "objectives": objectives,
            "direct_orders": self._direct_orders(board, analysis, stance, dominant),
            "constraints": constraints,
        }

    def simulate_llm(self, game_view: dict) -> dict:
        return self.decide(game_view)

    def _constraints(self, stance: str, dominant: bool) -> dict:
        garrison = int(self.profile["garrison"])
        commitment = int(self.profile["commitment"])
        if dominant:
            return {
                "min_general_garrison": max(1, garrison // 2),
                "min_city_garrison": 1,
                "max_commitment_percent": 100,
                "avoid_fog": False,
            }
        if stance == "defensive":
            return {
                "min_general_garrison": max(15, garrison),
                "min_city_garrison": 5,
                "max_commitment_percent": max(35, int(commitment * 0.75)),
                "avoid_fog": False,
            }
        return {
            "min_general_garrison": garrison,
            "min_city_garrison": 2,
            "max_commitment_percent": commitment,
            "avoid_fog": False,
        }

    def _analyze_board(self, board: Board) -> dict[str, list[tuple[int, int, Tile]]]:
        own_tiles: list[tuple[int, int, Tile]] = []
        enemy_tiles: list[tuple[int, int, Tile]] = []
        neutral_tiles: list[tuple[int, int, Tile]] = []
        fog_tiles: list[tuple[int, int, Tile]] = []
        frontier_tiles: list[tuple[int, int, Tile]] = []

        for y, row in enumerate(board.tiles):
            for x, tile in enumerate(row):
                if tile.occupier == self.player_idx:
                    own_tiles.append((x, y, tile))
                elif tile.occupier is not None and tile.occupier >= 0:
                    enemy_tiles.append((x, y, tile))
                elif tile.occupier == TILE_FOG:
                    fog_tiles.append((x, y, tile))
                elif tile.occupier is None and tile.type != TileType.MOUNTAIN:
                    neutral_tiles.append((x, y, tile))

        for x, y, tile in own_tiles:
            if any(self._is_actionable_neighbor(board.tiles[ny][nx]) for nx, ny in self._neighbors(board, x, y)):
                frontier_tiles.append((x, y, tile))

        return {
            "own_tiles": own_tiles,
            "enemy_tiles": enemy_tiles,
            "neutral_tiles": neutral_tiles,
            "fog_tiles": fog_tiles,
            "frontier_tiles": frontier_tiles,
        }

    def _direct_orders(
        self,
        board: Board,
        analysis: dict[str, list[tuple[int, int, Tile]]],
        stance: str,
        dominant: bool,
    ) -> list[dict]:
        sources = sorted(
            analysis["frontier_tiles"] or analysis["own_tiles"],
            key=lambda item: (-item[2].army, item[1], item[0]),
        )
        profile_orders = int(self.profile["direct_orders"])
        max_orders = profile_orders + 2 if dominant else profile_orders if stance == "aggressive" else max(1, profile_orders - 1)
        orders = []
        used_sources: set[tuple[int, int]] = set()
        used_targets: set[tuple[int, int]] = set()

        for x, y, tile in sources:
            if len(orders) >= max_orders:
                break
            if tile.army <= 1 or (x, y) in used_sources:
                continue
            step = self._best_adjacent_step(board, x, y, dominant)
            if step is None or step in used_targets:
                continue
            nx, ny = step
            orders.append({
                "from": {"x": x, "y": y},
                "to": {"x": nx, "y": ny},
                "amount": max(1, tile.army - 1),
            })
            used_sources.add((x, y))
            used_targets.add((nx, ny))
        return orders

    def _best_adjacent_step(self, board: Board, x: int, y: int, dominant: bool) -> tuple[int, int] | None:
        candidates: list[tuple[int, int, int, int]] = []
        for nx, ny in self._neighbors(board, x, y):
            tile = board.tiles[ny][nx]
            if tile.type == TileType.MOUNTAIN or tile.occupier == TILE_FOG_OBSTACLE:
                continue
            if tile.occupier == self.player_idx:
                continue
            rank = self._adjacent_rank(tile, dominant)
            candidates.append((rank, tile.army, ny, nx))
        if not candidates:
            return None
        _, _, ny, nx = min(candidates)
        return nx, ny

    def _adjacent_rank(self, tile: Tile, dominant: bool) -> int:
        if tile.occupier is not None and tile.occupier >= 0:
            if tile.type == TileType.GENERAL:
                return 0
            if tile.type == TileType.CITY:
                return 1
            return 2
        if tile.occupier is None and tile.type == TileType.CITY:
            return max(1, round(3 / float(self.profile["city_bias"])))
        if tile.occupier is None:
            return 4
        if tile.occupier == TILE_FOG:
            return max(2, round((5 if dominant else 6) / float(self.profile["explore_bias"])))
        return 9

    def _best_attack_target(self, analysis: dict[str, list[tuple[int, int, Tile]]]) -> tuple[int, int] | None:
        if not analysis["enemy_tiles"]:
            return None

        def score(item: tuple[int, int, Tile]) -> tuple[int, int, int, int]:
            x, y, tile = item
            if tile.type == TileType.GENERAL:
                type_rank = 0
            elif tile.type == TileType.CITY:
                type_rank = 1
            else:
                type_rank = 2
            return type_rank, tile.army, self._nearest_owned_distance(x, y, analysis), x + y

        x, y, _ = min(analysis["enemy_tiles"], key=score)
        return x, y

    def _best_pressure_target(
        self,
        board: Board,
        analysis: dict[str, list[tuple[int, int, Tile]]],
        stats: dict,
    ) -> tuple[int, int] | None:
        targets = analysis["fog_tiles"] or analysis["neutral_tiles"] or analysis["frontier_tiles"]
        if not targets:
            general = stats.get("general")
            if self._is_position(general):
                return int(general[0]), int(general[1])
            return None
        x, y, _ = min(
            targets,
            key=lambda item: (self._nearest_owned_distance(item[0], item[1], analysis), item[1], item[0]),
        )
        return x, y

    def _expansion_region(
        self,
        board: Board,
        stats: dict,
        analysis: dict[str, list[tuple[int, int, Tile]]],
        dominant: bool,
    ) -> dict:
        targets = analysis["enemy_tiles"] if dominant and analysis["enemy_tiles"] else []
        targets = targets or analysis["neutral_tiles"] or analysis["fog_tiles"] or analysis["frontier_tiles"] or analysis["own_tiles"]
        radius = 5 if dominant else 3
        return self._region_around_points(board, targets, radius, stats.get("general"))

    def _frontier_region(self, board: Board, stats: dict, analysis: dict[str, list[tuple[int, int, Tile]]]) -> dict:
        targets = analysis["frontier_tiles"] or analysis["enemy_tiles"] or analysis["own_tiles"]
        return self._region_around_points(board, targets, 3, stats.get("general"))

    def _defense_region(self, board: Board, stats: dict) -> dict:
        general = stats.get("general")
        if self._is_position(general):
            x, y = int(general[0]), int(general[1])
        else:
            x, y = board.width // 2, board.height // 2
        return self._bounded_region(board, x - 2, y - 2, x + 2, y + 2)

    def _region_around_points(
        self,
        board: Board,
        points: list[tuple[int, int, Tile]],
        radius: int,
        fallback_center: Any,
    ) -> dict:
        if points:
            x = round(sum(point[0] for point in points) / len(points))
            y = round(sum(point[1] for point in points) / len(points))
        elif self._is_position(fallback_center):
            x, y = int(fallback_center[0]), int(fallback_center[1])
        else:
            x, y = board.width // 2, board.height // 2
        return self._bounded_region(board, x - radius, y - radius, x + radius, y + radius)

    def _bounded_region(self, board: Board, x1: int, y1: int, x2: int, y2: int) -> dict:
        return {
            "x1": max(0, min(x1, board.width - 1)),
            "y1": max(0, min(y1, board.height - 1)),
            "x2": max(0, min(x2, board.width - 1)),
            "y2": max(0, min(y2, board.height - 1)),
        }

    def _nearest_owned_distance(
        self,
        x: int,
        y: int,
        analysis: dict[str, list[tuple[int, int, Tile]]],
    ) -> int:
        if not analysis["own_tiles"]:
            return 9999
        return min(abs(x - ox) + abs(y - oy) for ox, oy, _ in analysis["own_tiles"])

    def _is_actionable_neighbor(self, tile: Tile) -> bool:
        if tile.type == TileType.MOUNTAIN or tile.occupier == TILE_FOG_OBSTACLE:
            return False
        return tile.occupier != self.player_idx

    def _plan_text(self, stance: str, dominant: bool, alive_enemies: int, contact: bool) -> str:
        if dominant:
            return f"Heuristic: convert material lead into contact and captures against {alive_enemies} enemies."
        if contact:
            return "Heuristic: fight visible enemies while reinforcing the frontier."
        if stance == "defensive":
            return "Heuristic: stabilize the general, reinforce weak borders, and expand safely."
        return "Heuristic: expand toward cities and fog until reliable contact is made."

    def _neighbors(self, board: Board, x: int, y: int) -> list[tuple[int, int]]:
        candidates = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        return [(nx, ny) for nx, ny in candidates if board.in_bounds(nx, ny)]

    def _require_board(self, game_view: dict) -> Board:
        board = game_view.get("board")
        if not isinstance(board, Board):
            raise ValueError("game_view must contain a Board at key 'board'")
        return board

    def _is_position(self, position: Any) -> bool:
        return (
            isinstance(position, (tuple, list))
            and len(position) == 2
            and isinstance(position[0], int)
            and isinstance(position[1], int)
        )
