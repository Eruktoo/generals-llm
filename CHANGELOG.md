# Changelog

## Unreleased

### Added

- Added this README and current handoff documentation for context recovery in new sessions.
- Added map-generation safeguards so starts are not enclosed by mountains or neutral cities.
- Added terrain connectivity repair after map generation so passable non-city terrain is one connected component.
- Added opening spread execution so bots keep expanding during multi-half-turn strategic rounds instead of relying only on first-turn direct orders.
- Added recent-edge tactical memory for heuristic bots in both arena and 4-player main demo execution.
- Added explicit neutral city capture objectives once nearby force is sufficient.

### Changed

- Opening expansion now targets locally reachable frontier opportunities instead of being pulled toward globally visible neutral city tiles.
- Direct opening orders prefer outward frontier sources and avoid repeatedly feeding only from the general.
- Pathing and fallback expansion avoid treating neutral cities as cheap expansion targets.
- Main demo and arena now both clear stale direct orders on non-refresh turns while preserving current turn, stats, rankings, own tile count, and recent movement context.
- City handling now favors staging outside a city until force is sufficient, then capturing, instead of weak repeated city entries.

### Fixed

- Fixed 1v1 seed 9 where the conservative bot's bottom-right start could be sealed into a 4-tile pocket by mountains/cities.
- Fixed severe early-game idle behavior where bots could wait until turn 25 or later before meaningful expansion.
- Fixed immediate back-and-forth movement loops, especially conservative stacks moving between the general and adjacent defensive tiles.
- Fixed stale direct orders causing repeated opening moves from outdated strategic context.
- Fixed main demo divergence from arena execution logic.

### Validation Snapshot

- `python -m compileall engine agents execution arena web`
- `heuristic:crazy` vs `heuristic:conservative`, seed 9, 5 games: P0 4-1, finished 100%, average reverse move rate 0.0%, average zero half-turn rate 3.3%.
- `heuristic:baseline` vs `random`, seed 1, 5 games: P0 5-0, finished 100%.
- Seed 9 map after generation: both starts have 4 exits; passable terrain component is fully connected.
