# CellUniverse Project Memory

## User Preferences
- Maintain a changelog of all file changes with exact line numbers and before/after code blocks. See [changelog.md](changelog.md).
- Write changelog/memory to both `C++/examples/` and Claude's persistent memory directory.
- Do NOT attempt to build the project. User will build and test themselves.
- Update `C++/examples/details.md` when structural changes are made to the codebase.

## Project Structure
- C++ cell tracking simulation in `C++/`
- Config: `C++/examples/config.yaml`
- Main entry: `C++/src/main.cpp`
- Key files: `Lineage.cpp`, `Frame.cpp`, `Spheroid.cpp`, `ConfigTypes.hpp`
- Build: `C++/build/` (cmake)
- Branch: `jihang-spheroid`

## Implemented: Pre-optimization boundary preservation
When Phase 1 collapses an elongated/pancaked cell to a sphere, the PCA search boundary shrinks. Fix: save pre-optimization radii before Phase 1, pass to `getSplitCells`, use `max(current, preOpt)` for boundary so search area never shrinks below the pre-optimization size.

## Reverted: All three dim-cell brightness approaches
Three approaches tried and reverted: (1) perturbable `_intensity` — flattened cost surface, all cells grew to maxRadius; (2) analytically computed `_brightness` rendering — same 75% gradient drop; (3) real image normalization — weakened optimizer (50% fewer accepted perturbations), caused false splits with daughters stacking to mimic parent. The dim cell shrinkage problem remains open.

## Implemented: Daughter-daughter overlap threshold 0.5
Increased from 0.2 to prevent daughters from stacking in z to mimic parent. Legitimate splits (sep/radii ratio ~1.0) pass easily; parent-mimicking stacks (ratio ~0.2) get rejected. Applied in both initial overlap check and burn-in loop in `Frame.cpp`.

## Kept: Deep-clone fix for `_realFrameCopy`
`cv::Mat` copy constructor is shallow. Frame constructor now uses `.clone()` for `_realFrameCopy` so it has independent pixel data. This was needed for normalization but is a correctness fix regardless.
