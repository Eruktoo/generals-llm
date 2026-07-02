# Generals LLM — 技术实现规格

## 架构

```
game_engine/        ← Python 游戏引擎（核心逻辑）
agents/             ← LLM Agent 层
execution/          ← 战略→战术执行层
web/                ← 前端 SPA
```

## Phase 1: 游戏引擎 (Python)

引擎不需要任何外部依赖。纯 Python 3，标准库。

### 核心类

**`Tile`** — 一个格子
- type: enum(PLAIN, MOUNTAIN, CITY, GENERAL, SWAMP, DESERT)
- occupier: int | None (玩家索引或 None)
- army: int（部队数量）

**`Board`** — 地图
- width, height: int
- tiles: list[list[Tile]]
- 方法: get_visible(player_index) → 返回该玩家视野内的 BoardView

**`Game`** — 游戏
- board: Board
- players: list[PlayerState]
- turn: int
- phase: int (0 or 1, 半回合)
- alive: list[bool]
- generals: list[(x, y)]（每个存活的将军位置）
- cities: list[(x, y)]
- 方法:
  - `init(width, height, num_players, seed)` — 生成地图 + 放置玩家
  - `step()` — 推进一个半回合
  - `execute_moves(moves_dict)` — 接收 {player_idx: [Move, ...]}，执行
  - `get_player_view(player_idx)` — 返回该玩家的可见视图

**`Move`** — 移动指令
- from_x, from_y, to_x, to_y: int
- take_half: bool (True=分一半兵走，False=留1个走剩下的)

### 地图生成（简化版）

不追求 generals.io 同款复杂地图生成。简单的随机地图：

1. 生成网格（12×12），全部 PLAIN
2. 随机放置 MOUNTAIN（~15% 密度）
3. 随机放置 CITY（8-12 个）
4. 随机分配玩家起始位置（均匀分布，不靠近边界）
5. 玩家起始位置设为 GENERAL，初始兵力 0

### 战争迷雾

```
每个半回合后更新视野：
- 每个玩家的军队能看见自己所在格 + 相邻4格
- 城市提供自身+相邻4格视野
- 不可见的格子返回：TILE_FOG（迷雾）
- 迷雾中的障碍物（山/城市）返回 TILE_FOG_OBSTACLE
```

### 回合逻辑

```
每个 step() = 一个半回合：

Phase 0 (偶数半回合):
  - 执行收集到的所有移动
  - 产兵（城市/将军 +1，每 25 回合全体 +1）

Phase 1 (奇数半回合):
  - 执行收集到的所有移动
  - 不产兵

移动执行（同 generals.io 规则）：
  - 移动后来源格兵力: 如果 take_half → floor(原兵力/2)；否则 → 1
  - 目标格是友军 → 兵力叠加
  - 目标格是敌军 → 兵力相减，负了则占领（进攻将军则击杀）
  - 目标格是中立 → 兵力相减，负了则占领
```

### 验证

- 内置一个 `RandomBot` — 每回合随机合法移动
- 跑 `python3 -m engine.test` 验证 4 个 RandomBot 能正常完成一局

---

## Phase 2: 执行层 (Execution Layer)

把 LLM 的「战略目标」翻译成具体的 Move。

### 执行算法

输入：
- LLM 输出的目标列表 + 约束
- 当前 BoardView（该玩家视野）

输出：
- list[Move]（合法移动指令列表）

### 目标类型

| 目标类型 | 参数 | 行为 |
|---------|------|------|
| `expand_region` | x1,y1,x2,y2（区域）, priority | 向该区域的可见中立格推进 |
| `reinforce_region` | x1,y1,x2,y2, priority | 将周围兵力集中到该区域的前线 |
| `attack_position` | x,y, commitment(limited/full) | 集结兵力攻击指定位置 |
| `defend_region` | x1,y1,x2,y2, priority | 在区域内维持防线 |
| `direct_move` | from_x,from_y,to_x,to_y, amount | 精确指令（高优先级覆盖） |

### 约束

```python
{
    "min_general_garrison": 20,      # 将军格最少留多少兵
    "min_city_garrison": 5,          # 城市最少留多少兵
    "max_commitment_percent": 50,    # 单次进攻最多调用%总兵力
    "avoid_fog": False               # 是否避免向迷雾移动
}
```

### 优先级排序

如果 LLM 给多个目标，执行层按 priority 排序执行。资源（兵力）先满足高优先级。

---

## Phase 3: Agent 层

### 4 个 AI 人格

通过 system prompt 差异实现：

```python
AGENTS = {
    "gambler": {
        "system_prompt": "你是个激进的赌徒型指挥官...",
        "default_stance": "aggressive",
        "bluff_probability": 0.3,
    },
    "conservative": {
        "system_prompt": "你是保守型指挥官，注重防守...",
        "default_stance": "defensive",
    },
    "trickster": {
        "system_prompt": "你是个诡计多端的指挥官...",
        "default_stance": "balanced",
    },
    "crazy": {
        "system_prompt": "你的决策完全不可预测...",
        "default_stance": "random",
    },
}
```

### Prompt 输入模板

```
你是指挥官 [NAME]。[PERSONALITY]

当前回合: [TURN]
你还活着。敌人有 [N] 个存活。

你的军队:
- 总兵力: [TOTAL_ARMY]
- 占领格数: [TILES]
- 城市: [CITIES]
- 将军位置: (X, Y)
- 每分钟产兵: [PRODUCTION]

可见地图 (12×12):
[COMPACT_GRID]

前线局势:
- 与中立接壤: [N] 格 (推荐扩张方向)
- 与已知敌人接壤: [N] 格 (威胁等级 [LEVEL])
- 可见敌军总兵力: ~[ENEMY_ARMY]

上一轮你的战略意图: [PREV_INTENT]
执行结果: [EXECUTION_RESULT]

请输出你的战略指令（JSON）:
{
    "stance": "aggressive|defensive|balanced|random",
    "round_plan": "一句话说明本轮战略意图",
    "objectives": [
        {
            "type": "expand_region|reinforce_region|attack_position|defend_region",
            "region": {"x1": ..., "y1": ..., "x2": ..., "y2": ...},
            "target": {"x": ..., "y": ...},  # 仅 attack_position 需要
            "priority": 0.0-1.0,
            "commitment": "limited|full"  # 仅 attack 需要
        }
    ],
    "direct_orders": [  # 精确指令（可选）
        {"from": {"x": ..., "y": ...}, "to": {"x": ..., "y": ...}, "amount": N}
    ],
    "constraints": {
        "min_general_garrison": N,
        "avoid_fog": true|false
    }
}
```

### 输出验证

- 解析 JSON
- 校验字段完整性
- 如果解析失败，使用上次的目标 + 轻微随机偏移（容错）

---

## Phase 4: 前端

### 页面布局

```
┌──────────────────────────────────────────────┐
│  ⚔️ Generals LLM — 第 42 轮                   │
├──────────────────────────────────────────────┤
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │                                        │  │
│  │        12×12 网格战场                   │  │
│  │        Canvas 渲染                      │  │
│  │        迷雾 / 军队数字 / 颜色标识        │  │
│  │        移动动画                         │  │
│  │                                        │  │
│  └────────────────────────────────────────┘  │
│                                              │
├──────┬──────┬──────┬──────┬──────────────────┤
│ 🎲   │ 🎩   │ 🕵️   │ 🤡   │  游戏记录        │
│ 赌徒  │ 保守  │ 诡计  │ 疯子  │  第10轮:       │
│ 32兵  │ 28兵  │ 41兵  │ 19兵  │  赌徒攻击城市  │
│ 8地   │ 7地   │ 10地  │ 5地   │  诡计者扩张西北│
│ 激进  │ 防守  │ 均衡  │ ???   │  ...           │
└──────┴──────┴──────┴──────┴──────────────────┘
```

### 实现

- 单 HTML + Canvas + CSS
- WebSocket 连接后端获取游戏状态
- 动画：军队从 A→B 平滑移动
- 部署：`generals.shuttlescope.org` → Cloudflare Tunnel

---

## 构建顺序

```
1.  Game Engine (Python)       ← 现在从这里开始
2.  Execution Layer (Python)   ← 引擎验证后
3.  LLM Agent (Python)         ← 接 DeepSeek
4.  Web 前端                   ← 最后
5.  部署 + 调优               ← 修 prompt
```
