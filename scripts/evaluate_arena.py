from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arena import ArenaConfig, run_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 1v1 Generals bot evaluation matches.")
    parser.add_argument("--bots", nargs=2, default=["heuristic:baseline", "random"])
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument(
        "--strategic-interval",
        type=int,
        default=1,
        help="Half-turns between strategy recomputes for heuristic bots.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [args.seed + offset for offset in range(args.games)]
    summary = run_batch(
        args.bots,
        seeds,
        ArenaConfig(
            width=args.width,
            height=args.height,
            max_turns=args.max_turns,
            strategic_interval=args.strategic_interval,
        ),
    )

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print(f"matches: {summary['matches']}")
    print(f"bots: {summary['bots'][0]} vs {summary['bots'][1]}")
    print(f"wins: P0={summary['wins'][0]} P1={summary['wins'][1]} draws={summary['draws']}")
    print(f"win_rates: P0={summary['win_rates'][0]:.1%} P1={summary['win_rates'][1]:.1%}")
    print(f"finished_rate: {summary['finished_rate']:.1%}")
    print(f"avg_turns: {summary['avg_turns']}")
    print(f"avg_moves_per_half_turn: {summary['avg_moves_per_half_turn']}")
    print(f"avg_reverse_move_rate: {summary['avg_reverse_move_rate']:.1%}")
    print(f"avg_zero_half_turn_rate: {summary['avg_zero_half_turn_rate']:.1%}")
    print(f"dominance_lag_turns: {summary['dominance_lag_turns']}")


if __name__ == "__main__":
    main()
