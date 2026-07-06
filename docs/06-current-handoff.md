# Current Handoff

This document is the quickest way to resume work in a fresh conversation.

## Current Goal

The project is tuning heuristic Generals.io-style bots for more convincing openings and cleaner 1v1 arena behavior while keeping the 4-player main demo on the same core logic.

The latest work focused on three recurring problems:

- early city rushing or city-adjacent idling
- opening armies stuck near the general instead of spreading
- immediate back-and-forth movement loops, especially conservative defensive stacks

The latest known user preference is: make bots smarter through flow and evaluation, not through many rigid turn-count rules.

## Important Architecture

`agents/heuristic_agent.py` decides intent. It should express priorities and personality:

- `crazy`: more direct orders, more aggressive expansion/attack
- `conservative`: higher defense/garrison bias
- `gambler`, `trickster`, `baseline`: intermediate mixes

`execution/executor.py` owns tactical legality and sanity:

- never weakly enter neutral cities
- do opening spread flow
- avoid immediate reverse moves unless tactically justified
- stage city attacks until enough force is nearby
- execute visible general attacks and endgame hunts

`arena/bots.py` and `web/server.py` both pass recent movement edges into executor constraints. Keep these in sync whenever execution context changes.

`engine/board.py` owns map fairness. Current map generation:

- protects a radius-2 diamond around each start from mountains/cities
- places mountains and cities outside protected zones
- connects passable non-city terrain after generation
- then places generals

## Known Good Replays

Use these for regression:

```text
seed=2, heuristic:crazy vs heuristic:conservative, 12x12, max_turns=400, strategic_interval=1
```

Expected after latest fixes:

- both expand normally by turns 10/20
- no severe conservative general-adjacent stack bouncing
- early immediate reverse count should be near zero

```text
seed=9, heuristic:crazy vs heuristic:conservative, 12x12, max_turns=400, strategic_interval=1
```

This used to expose a map-generation bug where the conservative bottom-right start was sealed into 4 tiles. Expected now:

- both starts have 4 exits
- passable terrain is connected
- turn 10 should be roughly 9 tiles for both bots
- conservative is no longer trapped at 4 tiles

## Useful Commands

Compile:

```bash
python -m compileall engine agents execution arena web
```

Arena evaluation:

```bash
python scripts/evaluate_arena.py --games 5 --seed 9 --bots heuristic:crazy heuristic:conservative --width 12 --height 12 --max-turns 400 --strategic-interval 1
python scripts/evaluate_arena.py --games 5 --seed 1 --bots heuristic:baseline random --width 12 --height 12 --max-turns 400 --strategic-interval 1
```

Arena web:

```bash
python -B arena_web/server.py
```

Main web:

```bash
python -B web/server.py
```

## Deployment Notes

The public arena and main demo run from the same Python codebase on a remote server. Do not commit SSH keys, route commands, host-specific paths, or private operator notes. Keep those in local `AGENTS.md` or equivalent ignored files.

After deployment, check:

- arena process is serving `/api/replay`
- main process is serving `/api/status`
- seed 9 replay uses the repaired map

## Remaining Areas To Watch

- City timing is acceptable but still heuristic. User likes staging outside a city until enough force exists; they may later prefer stronger "expand first, city second" tempo.
- `baseline vs random` reverse rate can look high because random contributes many reverse moves. Inspect per-player metrics before interpreting it as heuristic jitter.
- The main demo can finish quickly once one bot snowballs; inspect early rounds separately from final winner.
- Map connectivity currently treats neutral cities as blockers for passable terrain. This is intentional for start fairness, but city density can still shape chokepoints.
