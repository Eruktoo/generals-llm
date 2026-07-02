from __future__ import annotations

import json
import os
import random
from typing import Any

from engine.board import TILE_FOG, TILE_FOG_OBSTACLE, Board
from engine.types import Tile, TileType


DEFAULT_CONSTRAINTS = {
    "min_general_garrison": 20,
    "min_city_garrison": 5,
    "max_commitment_percent": 50,
    "avoid_fog": False,
}


AGENT_CONFIGS = {
    "gambler": {
        "name": "赌徒",
        "emoji": "🎲",
        "system_prompt": """你是「赌徒」，一个激进、敢于冒险的指挥官。
你喜欢集中兵力打大仗。看到机会就all-in。
你讨厌保守和等待。宁可输得轰轰烈烈，也不要赢的窝窝囊囊。
你的座右铭：风险越大，回报越大。
你经常虚张声势，让对手以为你比实际更强。""",
        "default_stance": "aggressive",
        "default_constraints": {"min_general_garrison": 10, "max_commitment_percent": 60},
    },
    "conservative": {
        "name": "保守派",
        "emoji": "🎩",
        "system_prompt": """你是「保守派」，一个谨慎、稳健的指挥官。
你重视防御和可持续发展。扩张稳扎稳打，不冒进。
你讨厌赌博和风险。每次进攻前都要确保退路。
你的座右铭：先立于不败之地，再求胜。
你靠经济和防守取胜，不靠奇袭。""",
        "default_stance": "defensive",
        "default_constraints": {"min_general_garrison": 30, "max_commitment_percent": 30},
    },
    "trickster": {
        "name": "诡计者",
        "emoji": "🕵️",
        "system_prompt": """你是「诡计者」，一个精于计算、善于欺骗的指挥官。
你喜欢声东击西、诱敌深入。擅长制造假象。
你总是保留一支预备队，等对手露出破绽再致命一击。
你的座右铭：战争胜负在开战前就已决定。
你善于利用对手的心理，让他们做出错误判断。""",
        "default_stance": "balanced",
        "default_constraints": {"min_general_garrison": 20, "max_commitment_percent": 40},
    },
    "crazy": {
        "name": "疯子",
        "emoji": "🤡",
        "system_prompt": """你是「疯子」，一个完全不可预测的指挥官。
你的决策没有任何逻辑可言。有时候全力进攻，有时候全军撤退。
有时候把最强的兵放在最没用的地方。
你的对手永远猜不到你要做什么——因为你自己也不知道。
你的座右铭：如果敌人知道你要做什么，你已经输了。""",
        "default_stance": "balanced",
        "default_constraints": {"min_general_garrison": 5, "max_commitment_percent": 80},
    },
}


VALID_STANCES = {"aggressive", "defensive", "balanced"}
VALID_OBJECTIVE_TYPES = {
    "expand_region",
    "attack_position",
    "defend_region",
    "reinforce_region",
}
VALID_COMMITMENTS = {"limited", "full"}


class LLMAgent:
    DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
    DEEPSEEK_API_URL = "https://api.siliconflow.cn/v1/chat/completions"

    def __init__(self, player_idx: int, personality: str):
        if personality not in AGENT_CONFIGS:
            valid = ", ".join(sorted(AGENT_CONFIGS))
            raise ValueError(f"Unknown personality {personality!r}. Expected one of: {valid}")
        self.player_idx = player_idx
        self.personality = personality
        self.config = AGENT_CONFIGS[personality]
        self.history: list[tuple[str, str, dict]] = []
        self._last_game_view: dict | None = None

    def format_prompt(self, game_view: dict) -> str:
        board = self._require_board(game_view)
        stats = game_view.get("stats") or {}
        rankings = game_view.get("rankings") or []
        alive_enemies = sum(
            1
            for ranking in rankings
            if ranking.get("player") != self.player_idx and ranking.get("alive")
        )

        general = stats.get("general")
        general_text = self._format_position(general)
        production = self._production_per_turn(stats)

        lines = [
            self.config["system_prompt"],
            "",
            f"你是指挥官 {self.config['name']}。",
            "",
            f"当前回合: {int(game_view.get('turn', 0))}",
            f"当前阶段: {int(game_view.get('phase', 0))}",
            f"敌人存活: {alive_enemies}",
            "",
            "你的军力:",
            f"- 总兵力: {int(stats.get('army', 0))}",
            f"- 占领格数: {int(stats.get('tiles', 0))}",
            f"- 城市: {int(stats.get('cities', 0))}",
            f"- 将军位置: {general_text}",
            f"- 每回合产兵: {production}",
            "",
            f"战场态势（你视野内的地图，Map {board.width}x{board.height}）:",
            self._format_board(board),
            "",
            "图例: . = plain, M = mountain, C:n = neutral city, P.n = player tile, PG.n = general, PC.n = city, # = fog, ? = fog obstacle",
            "",
            "排名:",
            *self._format_rankings(rankings),
            "",
            "请输出你的战略指令（JSON格式）:",
            "{",
            '    "stance": "aggressive|defensive|balanced",',
            '    "round_plan": "一句话说明你的战术意图",',
            '    "objectives": [',
            '        {"type": "expand_region", "region": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}, "priority": 0.7},',
            '        {"type": "attack_position", "target": {"x": 8, "y": 4}, "priority": 0.9, "commitment": "limited"},',
            '        {"type": "defend_region", "region": {"x1": 0, "y1": 0, "x2": 3, "y2": 3}, "priority": 0.5},',
            '        {"type": "reinforce_region", "region": {"x1": 6, "y1": 3, "x2": 11, "y2": 7}, "priority": 0.6}',
            "    ],",
            '    "direct_orders": [',
            '        {"from": {"x": 1, "y": 1}, "to": {"x": 2, "y": 1}, "amount": 20}',
            "    ],",
            '    "constraints": {',
            '        "min_general_garrison": 20,',
            '        "min_city_garrison": 5,',
            '        "avoid_fog": false',
            "    }",
            "}",
            "",
            "IMPORTANT: Output ONLY valid JSON. No extra text.",
        ]
        return "\n".join(lines)

    def decide(self, game_view: dict) -> dict:
        prompt = self.format_prompt(game_view)
        self._last_game_view = game_view

        response_text = self._call_deepseek_api(prompt)
        strategy = self.parse_response(response_text)
        if strategy is None:
            strategy = self.simulate_llm(game_view)

        self.history.append((prompt, response_text, strategy))
        return strategy

    def _call_deepseek_api(self, prompt: str) -> str:
        import urllib.request

        key_path = os.path.expanduser("~/.hermes/credentials/deepseek.txt")
        try:
            with open(key_path) as f:
                api_key = f.read().strip()
        except (FileNotFoundError, IOError):
            return ""

        payload = json.dumps(
            {
                "model": self.DEEPSEEK_MODEL,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.8,
                "max_tokens": 1500,
                "stream": False,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            self.DEEPSEEK_API_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[DeepSeek API error] {e}")
            return ""

    def simulate_llm(self, game_view: dict) -> dict:
        board = self._require_board(game_view)
        stats = game_view.get("stats") or {}
        rankings = game_view.get("rankings") or []
        analysis = self._analyze_board(board)

        stance = self._choose_stance(stats, rankings, analysis)
        constraints = {
            **DEFAULT_CONSTRAINTS,
            **self.config["default_constraints"],
            "avoid_fog": stance == "defensive",
        }

        objectives: list[dict] = []
        attack_target = self._best_attack_target(analysis)
        expand_region = self._expansion_region(board, stats, analysis)
        defend_region = self._defense_region(board, stats)
        reinforce_region = self._reinforce_region(board, stats, analysis)

        if stance == "aggressive":
            if attack_target is not None:
                objectives.append(
                    {
                        "type": "attack_position",
                        "target": {"x": attack_target[0], "y": attack_target[1]},
                        "priority": 0.95,
                        "commitment": "full" if self.personality in {"gambler", "crazy"} else "limited",
                    }
                )
            objectives.append({"type": "expand_region", "region": expand_region, "priority": 0.7})
            objectives.append({"type": "reinforce_region", "region": reinforce_region, "priority": 0.45})
            plan = "集中主力寻找突破口，优先攻击已暴露的薄弱目标。"
        elif stance == "defensive":
            objectives.append({"type": "defend_region", "region": defend_region, "priority": 0.9})
            objectives.append({"type": "reinforce_region", "region": reinforce_region, "priority": 0.75})
            objectives.append({"type": "expand_region", "region": expand_region, "priority": 0.35})
            plan = "稳住将军和边境，用少量兵力继续安全扩张。"
        else:
            objectives.append({"type": "expand_region", "region": expand_region, "priority": 0.75})
            if attack_target is not None:
                objectives.append(
                    {
                        "type": "attack_position",
                        "target": {"x": attack_target[0], "y": attack_target[1]},
                        "priority": 0.65,
                        "commitment": "limited",
                    }
                )
            objectives.append({"type": "reinforce_region", "region": reinforce_region, "priority": 0.55})
            plan = "扩张与集结并行，等待更明确的进攻窗口。"

        direct_orders = self._direct_orders(board, analysis, stance)

        strategy = {
            "stance": stance,
            "round_plan": plan,
            "objectives": objectives,
            "direct_orders": direct_orders,
            "constraints": constraints,
        }
        return self._validate_strategy(strategy, board)

    def parse_response(self, response_text: str) -> dict | None:
        parsed = self._parse_json_object(response_text)
        if parsed is None:
            if self._last_game_view is not None:
                return self.simulate_llm(self._last_game_view)
            return None

        board = None
        if self._last_game_view is not None:
            board = self._last_game_view.get("board")
        if isinstance(board, Board):
            return self._validate_strategy(parsed, board)
        return self._validate_strategy(parsed, None)

    def _format_board(self, board: Board) -> str:
        return "\n".join(
            " ".join(self._format_tile(tile) for tile in row)
            for row in board.tiles
        )

    def _format_tile(self, tile: Tile) -> str:
        if tile.occupier == TILE_FOG:
            return "#"
        if tile.occupier == TILE_FOG_OBSTACLE:
            return "?"
        if tile.type == TileType.MOUNTAIN:
            return "M"
        if tile.occupier is None:
            if tile.type == TileType.CITY:
                return f"C:{tile.army}"
            return "."

        prefix = str(tile.occupier)
        if tile.type == TileType.GENERAL:
            prefix += "G"
        elif tile.type == TileType.CITY:
            prefix += "C"
        return f"{prefix}.{tile.army}"

    def _format_rankings(self, rankings: list[dict]) -> list[str]:
        if not rankings:
            return ["无排名信息"]
        lines = []
        for index, ranking in enumerate(rankings, start=1):
            player = int(ranking.get("player", -1))
            army = int(ranking.get("army", 0))
            tiles = int(ranking.get("tiles", 0))
            alive = "✅" if ranking.get("alive") else "❌"
            kills = int(ranking.get("kills", 0))
            lines.append(f"{index}. P{player} {army}兵 {tiles}格 {alive} kills={kills}")
        return lines

    def _choose_stance(self, stats: dict, rankings: list[dict], analysis: dict) -> str:
        if self.personality == "crazy":
            seed = (int(stats.get("army", 0)) * 31) + len(analysis["enemy_tiles"]) * 17
            return random.Random(seed).choice(["aggressive", "defensive", "balanced"])

        default = str(self.config["default_stance"])
        army = int(stats.get("army", 0))
        tiles = max(1, int(stats.get("tiles", 0)))
        density = army / tiles
        own_rank = self._own_rank(rankings)
        visible_enemy = bool(analysis["enemy_tiles"])

        if default == "aggressive":
            if visible_enemy or density >= 5 or own_rank <= 2:
                return "aggressive"
            return "balanced"
        if default == "defensive":
            if density < 4 or own_rank > 2 or visible_enemy:
                return "defensive"
            return "balanced"

        if visible_enemy and density >= 4:
            return "aggressive"
        if density < 3:
            return "defensive"
        return "balanced"

    def _analyze_board(self, board: Board) -> dict:
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
            for nx, ny in self._neighbors(board, x, y):
                neighbor = board.tiles[ny][nx]
                if neighbor.type == TileType.MOUNTAIN or neighbor.occupier == TILE_FOG_OBSTACLE:
                    continue
                if neighbor.occupier != self.player_idx:
                    frontier_tiles.append((x, y, tile))
                    break

        return {
            "own_tiles": own_tiles,
            "enemy_tiles": enemy_tiles,
            "neutral_tiles": neutral_tiles,
            "fog_tiles": fog_tiles,
            "frontier_tiles": frontier_tiles,
        }

    def _best_attack_target(self, analysis: dict) -> tuple[int, int] | None:
        enemy_tiles = analysis["enemy_tiles"]
        if not enemy_tiles:
            return None

        def score(item: tuple[int, int, Tile]) -> tuple[int, int, int]:
            x, y, tile = item
            type_rank = 0 if tile.type == TileType.GENERAL else 1 if tile.type == TileType.CITY else 2
            return (type_rank, tile.army, x + y)

        x, y, _ = min(enemy_tiles, key=score)
        return x, y

    def _expansion_region(self, board: Board, stats: dict, analysis: dict) -> dict:
        targets = analysis["neutral_tiles"] or analysis["fog_tiles"] or analysis["frontier_tiles"] or analysis["own_tiles"]
        return self._region_around_points(board, targets, radius=3, fallback_center=stats.get("general"))

    def _defense_region(self, board: Board, stats: dict) -> dict:
        general = stats.get("general")
        if self._is_position(general):
            x, y = int(general[0]), int(general[1])
        else:
            x, y = board.width // 2, board.height // 2
        return self._bounded_region(board, x - 2, y - 2, x + 2, y + 2)

    def _reinforce_region(self, board: Board, stats: dict, analysis: dict) -> dict:
        targets = analysis["frontier_tiles"] or analysis["own_tiles"]
        return self._region_around_points(board, targets, radius=2, fallback_center=stats.get("general"))

    def _direct_orders(self, board: Board, analysis: dict, stance: str) -> list[dict]:
        if stance == "defensive":
            return []

        sources = sorted(
            analysis["frontier_tiles"] or analysis["own_tiles"],
            key=lambda item: item[2].army,
            reverse=True,
        )
        for x, y, tile in sources:
            if tile.army <= 2:
                continue
            step = self._best_neighbor_step(board, x, y, stance)
            if step is None:
                continue
            nx, ny = step
            return [
                {
                    "from": {"x": x, "y": y},
                    "to": {"x": nx, "y": ny},
                    "amount": max(1, tile.army - 1),
                }
            ]
        return []

    def _best_neighbor_step(self, board: Board, x: int, y: int, stance: str) -> tuple[int, int] | None:
        candidates: list[tuple[int, int, int, int]] = []
        for nx, ny in self._neighbors(board, x, y):
            tile = board.tiles[ny][nx]
            if tile.type == TileType.MOUNTAIN or tile.occupier == TILE_FOG_OBSTACLE:
                continue
            if tile.occupier == self.player_idx:
                continue
            if tile.occupier is not None and tile.occupier >= 0:
                rank = 0
            elif tile.occupier == TILE_FOG:
                rank = 1 if stance == "aggressive" else 3
            elif tile.type == TileType.CITY:
                rank = 2
            else:
                rank = 1
            candidates.append((rank, tile.army, nx, ny))
        if not candidates:
            return None
        _, _, nx, ny = min(candidates)
        return nx, ny

    def _validate_strategy(self, strategy: Any, board: Board | None) -> dict:
        if not isinstance(strategy, dict):
            strategy = {}

        stance = strategy.get("stance")
        if stance not in VALID_STANCES:
            stance = self.config["default_stance"]

        objectives = [
            objective
            for objective in (
                self._validate_objective(raw_objective, board)
                for raw_objective in strategy.get("objectives") or []
            )
            if objective is not None
        ]
        if not objectives and board is not None:
            objectives = [
                {
                    "type": "expand_region",
                    "region": self._bounded_region(board, 0, 0, board.width - 1, board.height - 1),
                    "priority": 0.5,
                }
            ]

        return {
            "stance": stance,
            "round_plan": str(strategy.get("round_plan") or "根据当前视野执行稳健行动。"),
            "objectives": objectives,
            "direct_orders": self._validate_direct_orders(strategy.get("direct_orders"), board),
            "constraints": self._validate_constraints(strategy.get("constraints")),
        }

    def _validate_objective(self, objective: Any, board: Board | None) -> dict | None:
        if not isinstance(objective, dict) or objective.get("type") not in VALID_OBJECTIVE_TYPES:
            return None

        objective_type = objective["type"]
        validated: dict[str, Any] = {
            "type": objective_type,
            "priority": self._clamp_float(objective.get("priority", 0.5), 0.0, 1.0),
        }
        if objective_type == "attack_position":
            target = self._validate_position(objective.get("target"), board)
            if target is None:
                return None
            validated["target"] = target
            commitment = objective.get("commitment", "limited")
            validated["commitment"] = commitment if commitment in VALID_COMMITMENTS else "limited"
            return validated

        region = self._validate_region(objective.get("region"), board)
        if region is None:
            return None
        validated["region"] = region
        return validated

    def _validate_direct_orders(self, direct_orders: Any, board: Board | None) -> list[dict]:
        if not isinstance(direct_orders, list):
            return []
        validated = []
        for order in direct_orders:
            if not isinstance(order, dict):
                continue
            source = self._validate_position(order.get("from"), board)
            target = self._validate_position(order.get("to"), board)
            if source is None or target is None:
                continue
            amount = order.get("amount", 1)
            if not self._is_int(amount):
                amount = 1
            validated.append({"from": source, "to": target, "amount": max(1, int(amount))})
        return validated

    def _validate_constraints(self, constraints: Any) -> dict:
        merged = {**DEFAULT_CONSTRAINTS, **self.config["default_constraints"]}
        if isinstance(constraints, dict):
            merged.update(constraints)
        return {
            "min_general_garrison": max(0, int(merged.get("min_general_garrison", 0))),
            "min_city_garrison": max(0, int(merged.get("min_city_garrison", 0))),
            "max_commitment_percent": self._clamp_int(merged.get("max_commitment_percent", 50), 1, 100),
            "avoid_fog": bool(merged.get("avoid_fog", False)),
        }

    def _parse_json_object(self, response_text: str) -> dict | None:
        text = response_text.strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _region_around_points(
        self,
        board: Board,
        points: list[tuple[int, int, Tile]],
        radius: int,
        fallback_center: Any = None,
    ) -> dict:
        if points:
            x = round(sum(point[0] for point in points) / len(points))
            y = round(sum(point[1] for point in points) / len(points))
        elif self._is_position(fallback_center):
            x, y = int(fallback_center[0]), int(fallback_center[1])
        else:
            x, y = board.width // 2, board.height // 2
        return self._bounded_region(board, x - radius, y - radius, x + radius, y + radius)

    def _validate_region(self, region: Any, board: Board | None) -> dict | None:
        if not isinstance(region, dict):
            return None
        keys = ("x1", "y1", "x2", "y2")
        if not all(self._is_int(region.get(key)) for key in keys):
            return None
        x1 = min(int(region["x1"]), int(region["x2"]))
        x2 = max(int(region["x1"]), int(region["x2"]))
        y1 = min(int(region["y1"]), int(region["y2"]))
        y2 = max(int(region["y1"]), int(region["y2"]))
        if board is not None:
            return self._bounded_region(board, x1, y1, x2, y2)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def _bounded_region(self, board: Board, x1: int, y1: int, x2: int, y2: int) -> dict:
        return {
            "x1": max(0, min(x1, board.width - 1)),
            "y1": max(0, min(y1, board.height - 1)),
            "x2": max(0, min(x2, board.width - 1)),
            "y2": max(0, min(y2, board.height - 1)),
        }

    def _validate_position(self, position: Any, board: Board | None) -> dict | None:
        if not isinstance(position, dict):
            return None
        if not self._is_int(position.get("x")) or not self._is_int(position.get("y")):
            return None
        x = int(position["x"])
        y = int(position["y"])
        if board is not None and not board.in_bounds(x, y):
            return None
        return {"x": x, "y": y}

    def _require_board(self, game_view: dict) -> Board:
        board = game_view.get("board")
        if not isinstance(board, Board):
            raise ValueError("game_view must contain a Board at key 'board'")
        return board

    def _production_per_turn(self, stats: dict) -> int:
        if not stats.get("alive", True):
            return 0
        return int(stats.get("cities", 0)) + (1 if stats.get("general") is not None else 0)

    def _own_rank(self, rankings: list[dict]) -> int:
        for index, ranking in enumerate(rankings, start=1):
            if ranking.get("player") == self.player_idx:
                return index
        return len(rankings) + 1

    def _format_position(self, position: Any) -> str:
        if self._is_position(position):
            return f"({int(position[0])}, {int(position[1])})"
        return "未知"

    def _is_position(self, position: Any) -> bool:
        return (
            isinstance(position, (tuple, list))
            and len(position) == 2
            and self._is_int(position[0])
            and self._is_int(position[1])
        )

    def _neighbors(self, board: Board, x: int, y: int) -> list[tuple[int, int]]:
        candidates = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        return [(nx, ny) for nx, ny in candidates if board.in_bounds(nx, ny)]

    def _clamp_float(self, value: Any, low: float, high: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = low
        return max(low, min(high, number))

    def _clamp_int(self, value: Any, low: int, high: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = low
        return max(low, min(high, number))

    def _is_int(self, value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)
