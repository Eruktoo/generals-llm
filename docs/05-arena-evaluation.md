# Arena Evaluation

This project now has a small 1v1 evaluation harness separate from the web replay service.

## Why

Single replay inspection is useful, but it is noisy. A bot can look smart or foolish because of one seed, one spawn, or one city placement. The arena runner gives us repeatable seed pools and aggregate metrics so bot quality can be compared over batches.

## Entry Point

Run a quick baseline evaluation:

```bash
python scripts/evaluate_arena.py --games 20 --seed 1 --bots heuristic:baseline random
```

Run two heuristic personalities with slower high-level strategy refresh:

```bash
python scripts/evaluate_arena.py --games 10 --seed 100 --bots heuristic:crazy heuristic:conservative --strategic-interval 12
```

Use the web-sized board when you want a closer match to the viewer, but expect it to be slower:

```bash
python scripts/evaluate_arena.py --games 3 --seed 200 --width 18 --height 18 --max-turns 300 --bots heuristic:crazy heuristic:conservative --strategic-interval 12
```

## Bot Specs

- `random`
- `heuristic:baseline`
- `heuristic:gambler`
- `heuristic:conservative`
- `heuristic:trickster`
- `heuristic:crazy`

The harness currently supports exactly two players. That is intentional: 1v1 is easier to evaluate, easier to watch, and better for detecting whether a bot can convert advantage into a win.

## Current Metrics

- `wins`, `draws`, `win_rates`
- `finished_rate`
- `avg_turns`
- `avg_moves_per_half_turn`
- `avg_reverse_move_rate`
- `avg_zero_half_turn_rate`
- `dominance_lag_turns`

`dominance_lag_turns` measures how long a player stayed materially ahead after first crossing a simple army/tile ratio threshold. High values usually mean the bot has a conversion problem: it can become stronger without ending the game.

## Early Read

On a 12x12 quick run, `heuristic:baseline` can beat `random`, which is the minimum sanity check.

On 18x18 with `heuristic:crazy` vs `heuristic:conservative`, short batches still tend to hit the turn limit. That matches recent replay observations: the opening is better than before, but midgame and endgame conversion are still weak.

The next useful step is not more prompt tuning. It is a stronger tactical planner with stable objectives, target locking, and a batch report page/API powered by this harness.
