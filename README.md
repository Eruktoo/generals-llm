# generals-llm

A simplified Generals.io-style AI arena for testing strategic bot behavior.

The project contains a pure Python game engine, heuristic personalities, a tactical executor, a 1v1 arena evaluator with replay UI, and a 4-player web demo. The current focus is making the heuristic bots play coherent openings, transition into city captures, and avoid tactical jitter such as immediate move reversals.

## Project Layout

- `engine/`: board generation, visibility, movement, combat, production, player state.
- `agents/`: strategic agents and heuristic personality profiles.
- `execution/`: converts strategy objects into legal moves with tactical safeguards.
- `arena/`: 1v1 batch runner, metrics, replay frame generation.
- `arena_web/`: local 1v1 evaluation and replay web UI, default port `8910`.
- `web/`: 4-player main demo, default port `8900`.
- `scripts/`: CLI evaluation helpers.
- `docs/`: design notes, retrospectives, evaluation notes, and current handoff.

## Quick Start

Run a 1v1 CLI evaluation:

```bash
python scripts/evaluate_arena.py --games 5 --seed 9 --bots heuristic:crazy heuristic:conservative --width 12 --height 12 --max-turns 400 --strategic-interval 1
```

Start the arena web UI:

```bash
python -B arena_web/server.py
```

Start the 4-player main demo:

```bash
python -B web/server.py
```

Useful replay URL shape:

```text
http://localhost:8910/replay?seed=9&bot0=heuristic%3Acrazy&bot1=heuristic%3Aconservative&width=12&height=12&max_turns=400&strategic_interval=1
```

## Current Bot Model

`HeuristicAgent` produces a strategy with:

- stance and explanatory reasoning
- objectives such as expansion, city capture, attack, defense, reinforcement
- direct opening orders
- constraints including opening state, garrison, commitment, and recent move context

`Executor` applies tactical execution in priority order:

1. visible general decapitation
2. endgame hunt when materially superior
3. direct orders
4. opening spread flow
5. strategic objectives
6. safe expansion fallback

The executor is intentionally responsible for move legality, weak city-entry prevention, opening flow, and anti-reversal behavior. Personality should affect preferences, not basic tactical sanity.

## Current Quality Checks

Recent useful checks:

```bash
python -m compileall engine agents execution arena web
python scripts/evaluate_arena.py --games 5 --seed 9 --bots heuristic:crazy heuristic:conservative --width 12 --height 12 --max-turns 400 --strategic-interval 1
python scripts/evaluate_arena.py --games 5 --seed 1 --bots heuristic:baseline random --width 12 --height 12 --max-turns 400 --strategic-interval 1
```

When investigating a replay, reproduce it locally with the same seed, bot specs, map size, max turns, and strategic interval. Inspect frame rankings at turns 10/20/30 and count immediate reverse moves for early frames.

## Operational Notes

The code is also deployed on a remote server for the public arena and main demo. Server hostnames, SSH keys, routes, and other machine-specific details should live in local operator notes such as `AGENTS.md`, not in this repository.
