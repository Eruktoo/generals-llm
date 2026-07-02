# Generals LLM — Critical Analysis

This design has a strong core idea: multiple LLM agents with private information can create genuine uncertainty, and a visible board game gives spectators a concrete way to judge decisions. The largest risk is not technical feasibility. The largest risk is that the LLM decision loop may sit at the wrong level of abstraction: too slow and sparse for tactical play, but too detailed and repetitive for strategic play.

## 1. Pacing

The proposed cadence is one LLM decision round every roughly 6-10 half-turns. That is probably playable, but it changes the game into something closer to "periodic war planning" than generals.io-style maneuvering.

The main question is whether the board state changes enough between LLM calls to make each round feel meaningful. In early-game expansion, 10 half-turns can be meaningful because cities, fog reveal, and frontier growth change the map. In mid-game positional fights, however, 10 half-turns can produce several local tactical opportunities that the LLM cannot react to. A doomed attack may continue too long. A vulnerable tile may be missed until the next round. A newly revealed enemy general or city may arrive too late for the current plan.

The bigger spectator risk is repetitiveness. If each round asks every agent for a full map move list, the LLM may repeatedly emit the same pattern:

- reinforce front
- expand into neutral tiles
- avoid overextending
- probe enemy
- consolidate armies

Those are reasonable plans, but they are not necessarily dramatic. The game needs visible decision moments: attacks, retreats, forks, bait, city captures, failed invasions, emergency defenses. If the engine silently auto-plays too many small steps between decisions, the audience may perceive the LLM as detached from the interesting events.

Recommendation: start with a shorter cadence, probably 4-6 half-turns, then tune upward only if the game feels too stop-start. Also consider variable pacing:

- Early game: 8-10 half-turns per decision, because expansion is low-risk and repetitive.
- First contact: 4-6 half-turns, because scouting and border formation matter.
- Active combat near enemy-owned tiles: 2-4 half-turns, because tactics matter.
- Late game: 4-6 half-turns, unless only cleanup remains.

The round should also have an explicit "intent" layer. Instead of merely issuing movement commands, each agent should state its current objective, such as "expand northwest", "hold eastern choke", "prepare city assault", or "retreat to capital". This makes repeated move sets easier to evaluate and gives the next prompt useful continuity.

## 2. Full LLM Decision

Having each AI output all unit moves in one call is likely too much if the board grows to 20x20 with 30+ owned tiles. It is not impossible for an LLM to emit a large move list, but the quality will be uneven. The model will usually handle high-level priorities better than tile-by-tile logistics.

The failure pattern is predictable:

- It will focus on salient areas and ignore quiet but important tiles.
- It may give illegal or contradictory moves from the same source tile.
- It may move armies out of defensive anchors by accident.
- It may miss small tactical wins because the board encoding is dense.
- It may produce plausible-looking but strategically incoherent move lists.

A 20x20 grid is not large for software, but it is large for a single language-model reasoning pass if the model must inspect every owned tile, infer local tactics, preserve long-term strategy, and output exact legal moves. The problem gets worse under fog because incomplete information requires uncertainty management too.

The better architecture is probably a strategy layer plus an execution layer:

1. LLM strategy layer chooses objectives, priorities, risk posture, and target regions.
2. A deterministic execution layer converts those objectives into legal moves.
3. The engine validates, clips, or rejects illegal moves.
4. The next LLM prompt reports what was actually executed and what changed.

For example, the LLM should be able to say:

```json
{
  "stance": "balanced",
  "objectives": [
    {
      "type": "expand",
      "region": "northwest frontier",
      "priority": 0.7
    },
    {
      "type": "reinforce",
      "region": "eastern border",
      "priority": 0.9
    },
    {
      "type": "attack",
      "target": {"x": 11, "y": 8},
      "priority": 0.6,
      "commitment": "limited"
    }
  ],
  "constraints": {
    "min_capital_garrison": 20,
    "avoid_emptying_border_tiles": true
  }
}
```

Then a simple algorithm can route armies, merge stacks, preserve garrisons, expand to neutrals, and avoid illegal duplicate orders. This also makes the game easier to tune. If the AI seems passive, adjust objective scoring. If it suicides, adjust commitment and safety constraints. If it fails tactically, improve the execution algorithm without changing the prompt.

There is still room for direct LLM move commands, but they should be reserved for a small number of high-impact overrides:

- "move main stack from A to B"
- "retreat this army"
- "attack this revealed city"
- "block this choke"

The LLM should not have to micromanage every farm tile.

## 3. Prompt Design

The prompt should be structured around decision quality, legality, and continuity. It should not be a raw dump of the entire board plus a vague request for moves.

Good prompt inputs:

- Rules summary: movement, combat, growth, fog, win/loss conditions.
- Current turn and phase: early expansion, first contact, mid-game, late-game if known.
- Player status: owned tile count, total army, cities, general location, production rate.
- Visible map: compact grid with terrain, owner, armies, cities, generals, fog.
- Frontier summary: owned tiles adjacent to neutral/enemy/fog.
- Threat summary: visible enemy stacks, likely attack paths, exposed important tiles.
- Opportunity summary: weak neutrals, cities, enemy overextensions, chokes.
- Previous intent: last round's objectives and whether they succeeded.
- Executed result: what the engine actually did from the last command set.
- Output schema: strict JSON with enumerated action types.

The model should receive both a grid and derived summaries. The grid preserves truth; the summaries direct attention. Without summaries, the LLM may miss details. Without the grid, it may overtrust the summarizer and make brittle decisions.

The prompt should also force tradeoffs. A weak prompt says "play well and output moves." A stronger prompt asks the agent to choose a risk posture and allocate effort:

- `economy`: expansion and city capture
- `defense`: capital safety, border garrisons, choke control
- `offense`: attacks, raids, pressure
- `scouting`: fog reveal and enemy localization

The output format should be strict JSON, not natural language. Natural language can be included only in a short `rationale` field for debugging and spectator display. The actual executable portion should be schema-validated.

Example output shape:

```json
{
  "risk": "balanced",
  "round_plan": "Hold the east while expanding through the northwest fog.",
  "priorities": {
    "economy": 0.35,
    "defense": 0.35,
    "offense": 0.15,
    "scouting": 0.15
  },
  "objectives": [
    {
      "id": "obj-1",
      "type": "reinforce_region",
      "region": {"x1": 8, "y1": 5, "x2": 11, "y2": 9},
      "priority": 0.9,
      "duration_half_turns": 6
    },
    {
      "id": "obj-2",
      "type": "expand_region",
      "region": {"x1": 2, "y1": 1, "x2": 6, "y2": 5},
      "priority": 0.7,
      "duration_half_turns": 6
    }
  ],
  "direct_orders": [
    {
      "from": {"x": 7, "y": 6},
      "to": {"x": 8, "y": 6},
      "amount": 18,
      "reason": "block visible enemy stack"
    }
  ],
  "constraints": {
    "min_general_garrison": 25,
    "max_army_commitment_percent": 45
  }
}
```

To avoid passivity, the prompt should reward tempo. It should explicitly say that doing nothing loses to expansion and that controlled risk is required. The scoring rubric can include:

- capture neutral production when safe
- keep armies moving toward useful frontiers
- convert army advantage into territory
- avoid leaving the general vulnerable
- attack only when local force ratio and follow-up are plausible

To avoid over-aggression, the model needs hard constraints and visible consequences:

- minimum general garrison
- maximum commitment to unknown fog
- do not attack cities without enough surplus
- do not split the main stack into many weak attacks
- retreat when local enemy force exceeds threshold

The prompt should ask for a plan under uncertainty, not fake certainty. The agent should be allowed to mark assumptions:

```json
"assumptions": [
  "Enemy red likely controls the southeast based on last seen border.",
  "Northwest fog is probably neutral because no enemy has appeared there."
]
```

This makes bad decisions easier to diagnose.

## 4. Failure Modes

The game can become boring if most rounds are low-information expansion with similar decisions. Four AIs may all expand quietly for many rounds, then collide in a slow border grind. If the viewer cannot understand why a move happened, the game becomes a screensaver rather than a strategy match.

It can become broken if LLM output is treated as authoritative. Illegal moves, duplicate source orders, stale coordinates, and inconsistent army counts should be expected. The engine needs validation and a fallback policy. Bad orders should degrade gracefully into partial execution, not crash the match or freeze an agent.

It can become uninteractive if the user has no meaningful role. "Watch four bots play" is interesting only if the bots produce recognizable personalities, readable plans, and surprising outcomes. Otherwise, the user may feel detached. Spectator affordances matter: show each agent's last plan, confidence, known map, and key assumptions. Let the user inspect why an attack happened.

Important failure modes:

- Strategic sameness: all agents receive similar prompts and converge on the same cautious style.
- Prompt drift: long-running memory makes agents repeat stale plans after the board changes.
- Tactical blindness: agents miss obvious local captures, retreats, or forks.
- Overplanning: agents emit detailed plans that are invalid after two half-turns.
- Underplanning: agents emit vague objectives that the execution layer interprets too generically.
- Fog confusion: agents treat unknown tiles as safe or assume invisible enemies are not there.
- Snowballing: one early city or general capture decides the game too quickly.
- Stalemate: agents overdefend borders and never commit enough force to break through.
- Latency drag: 8-second rounds feel acceptable early but slow when the outcome is obvious.
- Debuggability gap: when an agent plays badly, it is unclear whether the prompt, model, parser, or executor caused it.

The design should include mitigation from the beginning:

- schema validation
- deterministic fallback bot
- replay logs with prompt, output, parsed actions, executed actions, and board diff
- agent personality parameters
- adjustable aggression/risk weights
- shorter decision cadence during combat
- spectator-visible plans
- surrender or fast-forward logic for decided games

## 5. Better Alternatives

"Slow generals" is viable, but it has a fundamental tension: the original game is tactically dense and timing-sensitive, while LLMs are better at slower strategic decisions, negotiation, deception, and prioritization. If the goal is to showcase multi-agent LLM decision-making, there may be better fits.

Better candidates:

### Diplomacy-lite area control

A simplified Diplomacy-style map may fit LLMs better than generals.io. Agents issue simultaneous orders, negotiate or infer intent, form temporary alliances, and resolve moves deterministically. LLM strengths matter: persuasion, betrayal, long-term planning, reading opponents, and explaining intent. The board is smaller, actions are discrete, and each turn is meaningful.

Downside: implementing good negotiation UX is more complex, and games may run long.

### Small-grid tactics with few units

A tactics game with 3-6 units per agent on a 10x10 grid may be better than 30+ owned tiles. Each LLM call can reason about every unit in detail. Actions are understandable: move, attack, defend, capture, use ability. Spectators can see tactical causality immediately.

Downside: requires careful unit/rules design, and combat balance matters more.

### Simultaneous hidden-order naval game

A Battleship-plus-strategy game could work well: fleets move under partial information, agents choose scouting, attacking, retreating, and deception. The state is smaller than generals, fog is central, and simultaneous turns create suspense.

Downside: less visually rich unless the rules add territory, objectives, or resources.

### Economic territory game

A compact 4-player game where agents choose build, expand, attack, scout, or fortify each turn may better align with LLM planning. Instead of moving dozens of armies, agents allocate resources to regions. The game becomes about priorities and tradeoffs, not pathfinding.

Downside: it is less directly inspired by generals.io and may feel more abstract.

### Social deduction with board state

A visible board plus hidden roles or private goals can exploit LLM personality and deception. Agents must infer who is aligned with whom while still taking concrete map actions.

Downside: harder to evaluate objectively, and prompt leakage must be controlled carefully.

## Bottom Line

The current plan can work as an experiment, especially on a 12x12 board, but a full "LLM outputs every move for every owned tile" approach is likely the wrong long-term control model. The more robust design is:

- LLM decides strategy, objectives, posture, and a few critical direct orders.
- Code handles legal tactical execution, routing, garrisons, and repetitive expansion.
- Decision cadence adapts to game phase and local combat intensity.
- Prompts include both raw board state and derived tactical summaries.
- The frontend shows agent plans and assumptions, not just board animation.

If the project keeps the generals-like premise, it should lean into being a slow strategic war room rather than a delayed micro game. If that still feels thin after prototyping, a smaller-unit tactics game or Diplomacy-lite area-control game would likely showcase LLM agents more naturally.
