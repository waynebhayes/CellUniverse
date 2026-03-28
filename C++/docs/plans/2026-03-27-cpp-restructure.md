# C++ Directory Restructuring Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up the C++ directory by deprecating unused code, flattening the source tree, removing the vestigial Cell abstract base class include, cleaning dead method signatures, cleaning up the include graph, reorganizing examples/ into logical subdirectories, and modernizing CMakeLists.txt.

**Architecture:** Move all dead/unused files to `C++/deprecated/` (Bacilli, Sphere, Cell abstract base, mathhelper, args.cpp, pseudo-code.txt, stray TIF). Flatten `src/Cells/` since only Spheroid remains. Remove dead `#include "Cell.hpp"` from all active files and fix two Frame.hpp method signatures that take an unused `const Cell &` parameter. Reorganize `examples/` into `data/`, `config/`, `scripts/` subdirectories. Update CMakeLists.txt to match new layout.

**Constraints:**

- Do NOT delete any files — move to `C++/deprecated/` instead
- Do NOT touch `Python/` directory
- Do NOT build or run the project — user will verify
- Leave `C++/lib/` (yaml-cpp) and `C++/tests/` as-is
- Write a changelog entry after all changes

**Verified against actual code on 2026-03-27.** Key finding: Spheroid already does NOT inherit Cell (class declaration is `class Spheroid {`). Cell.hpp is only included for transitive access to headers, but `CellParams` (which SpheroidParams inherits) is defined in ConfigTypes.hpp, not Cell.hpp.

---

## Current C++ Directory Layout

```text
C++/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── args.cpp                          # NOT compiled, uses undefined cxxopts library
├── pseudo-code.txt                   # Outdated pseudocode
├── frame000.tif                      # Stray corrupted test file (0x0 dimensions)
├── includes/
│   ├── Bacilli.hpp                   # DEAD — never compiled, incomplete
│   ├── Cell.hpp                      # VESTIGIAL — abstract base, never used polymorphically
│   ├── CellFactory.hpp               # ACTIVE — includes Cell.hpp (unnecessary)
│   ├── ConfigTypes.hpp               # ACTIVE — defines CellParams, CellConfig, all config structs
│   ├── Frame.hpp                     # ACTIVE — includes Cell.hpp + Sphere.hpp (both unnecessary)
│   ├── Lineage.hpp                   # ACTIVE — includes Cell.hpp (unnecessary)
│   ├── LineageViewer.hpp             # ACTIVE (but viewer.update() commented out in main.cpp)
│   ├── mathhelper.hpp                # ORPHANED — included by nothing
│   ├── Sphere.hpp                    # DEAD — compiled but never instantiated
│   ├── Spheroid.hpp                  # ACTIVE — includes Cell.hpp (unnecessary, does NOT inherit Cell)
│   └── types.hpp                     # ACTIVE
├── src/
│   ├── main.cpp                      # ACTIVE
│   ├── CellFactory.cpp               # ACTIVE
│   ├── Frame.cpp                     # ACTIVE — has 2 methods with unused `const Cell &` param
│   ├── Lineage.cpp                   # ACTIVE
│   ├── LineageViewer.cpp             # ACTIVE (compiled, viewer.update() call commented out)
│   └── Cells/
│       ├── Bacilli.cpp               # DEAD — not in CMakeLists
│       ├── Sphere.cpp                # DEAD — compiled but never called
│       └── Spheroid.cpp              # ACTIVE
├── examples/
│   ├── config.yaml                   # Config
│   ├── initial.csv                   # Config
│   ├── initial_auto.csv              # Config
│   ├── initial_embryo.csv            # Config
│   ├── user_input_configurations.ini # Config
│   ├── runauto.args                  # Config
│   ├── run_celluniverse.sh           # Script
│   ├── run_original.sh               # Script
│   ├── run_embryo.sh                 # Script
│   ├── runauto.sh                    # Script
│   ├── remove_old_outputs.sh         # Script
│   ├── convert_png_to_tiff.py        # Script
│   ├── details.md                    # Documentation (belongs in docs/)
│   └── input/
│       └── original_data/            # 41 TIFF frames (~577 MB)
├── docs/                             # Changelogs, technical docs
├── tests/                            # Google Test (minimal)
├── lib/                              # yaml-cpp (bundled dependency)
└── build/                            # CMake build artifacts (regenerable)
```

## Target C++ Directory Layout

```text
C++/
├── CMakeLists.txt                    # UPDATED — new source paths, cleaned include dirs
├── README.md                         # UPDATED — new run instructions
├── .gitignore
├── includes/
│   ├── CellFactory.hpp               # UPDATED — removed #include "Cell.hpp"
│   ├── ConfigTypes.hpp               # UNCHANGED (still defines CellParams, CellConfig)
│   ├── Frame.hpp                     # UPDATED — removed Cell.hpp + Sphere.hpp includes,
│   │                                 #   fixed 2 method signatures, removed gradientDescent decl
│   ├── Lineage.hpp                   # UPDATED — removed #include "Cell.hpp"
│   ├── LineageViewer.hpp             # UNCHANGED
│   ├── Spheroid.hpp                  # UPDATED — removed #include "Cell.hpp"
│   └── types.hpp                     # UNCHANGED
├── src/
│   ├── main.cpp                      # UNCHANGED
│   ├── CellFactory.cpp               # UNCHANGED
│   ├── Frame.cpp                     # UPDATED — fixed 2 method signatures, removed dead
│   │                                 #   gradientDescent() commented block
│   ├── Lineage.cpp                   # UNCHANGED
│   ├── LineageViewer.cpp             # UNCHANGED
│   └── Spheroid.cpp                  # MOVED from src/Cells/, UPDATED include path
├── config/
│   ├── config.yaml                   # MOVED from examples/
│   ├── initial.csv                   # MOVED from examples/
│   ├── initial_auto.csv              # MOVED from examples/
│   ├── initial_embryo.csv            # MOVED from examples/
│   ├── user_input_configurations.ini # MOVED from examples/
│   └── runauto.args                  # MOVED from examples/
├── scripts/
│   ├── run_celluniverse.sh           # MOVED from examples/, UPDATED paths
│   ├── run_original.sh               # MOVED from examples/, UPDATED paths
│   ├── run_embryo.sh                 # MOVED from examples/, UPDATED paths
│   ├── runauto.sh                    # MOVED from examples/, UPDATED paths
│   ├── remove_old_outputs.sh         # MOVED from examples/
│   └── convert_png_to_tiff.py        # MOVED from examples/
├── data/
│   └── input/
│       └── original_data/            # MOVED from examples/input/
├── deprecated/
│   ├── README.md                     # Explains what's here and why
│   ├── Bacilli.hpp                   # From includes/
│   ├── Bacilli.cpp                   # From src/Cells/
│   ├── Sphere.hpp                    # From includes/
│   ├── Sphere.cpp                    # From src/Cells/
│   ├── Cell.hpp                      # From includes/ (vestigial abstract base)
│   ├── mathhelper.hpp                # From includes/ (orphaned)
│   ├── args.cpp                      # From C++/ root (not compiled)
│   ├── pseudo-code.txt               # From C++/ root (outdated)
│   └── frame000.tif                  # From C++/ root (corrupted stray)
├── docs/
│   ├── details.md                    # MOVED from examples/
│   ├── changelogv1.md
│   ├── changelogv2.md
│   ├── changelogv3_after_brightness_fix.md
│   └── ...                           # Other existing docs
├── tests/                            # UNCHANGED
├── lib/                              # UNCHANGED
└── build/                            # UNCHANGED (will need rebuild)
```

---

## Task 1: Create deprecated/ directory and move dead files

**Files:**

- Create: `C++/deprecated/README.md`
- Move: 9 files from various locations to `C++/deprecated/`

- [ ] **Step 1: Create deprecated directory with README**

```bash
mkdir -p C++/deprecated
```

Write `C++/deprecated/README.md`:

```markdown
# Deprecated Files

Files in this directory were removed from the active build on 2026-03-27 during
the C++ directory restructuring. They are preserved for historical reference.

## Contents

| File | Original Location | Reason |
|------|-------------------|--------|
| Bacilli.hpp | includes/ | Dead code — never compiled, incomplete implementation |
| Bacilli.cpp | src/Cells/ | Dead code — not in CMakeLists.txt, references undefined Rectangle class |
| Sphere.hpp | includes/ | Compiled but never instantiated in execution path |
| Sphere.cpp | src/Cells/ | Compiled but never called — CellFactory only creates Spheroid |
| Cell.hpp | includes/ | Vestigial abstract base — no class inherits it on this branch, no polymorphic usage |
| mathhelper.hpp | includes/ | Orphaned — not included by any file in the project |
| args.cpp | C++/ root | Not compiled — uses undefined cxxopts library |
| pseudo-code.txt | C++/ root | Outdated pseudocode, does not reflect current algorithm |
| frame000.tif | C++/ root | Stray test file with corrupted dimensions (0x0) |

## Restoring Files

If any of these files are needed again, copy them back to their original location
and update CMakeLists.txt accordingly.
```

- [ ] **Step 2: Move dead files using git mv**

```bash
cd C++

# Dead cell models
git mv includes/Bacilli.hpp deprecated/Bacilli.hpp
git mv src/Cells/Bacilli.cpp deprecated/Bacilli.cpp
git mv includes/Sphere.hpp deprecated/Sphere.hpp
git mv src/Cells/Sphere.cpp deprecated/Sphere.cpp

# Vestigial abstract base
git mv includes/Cell.hpp deprecated/Cell.hpp

# Orphaned header
git mv includes/mathhelper.hpp deprecated/mathhelper.hpp

# Root-level stray files
git mv args.cpp deprecated/args.cpp
git mv pseudo-code.txt deprecated/pseudo-code.txt
git mv frame000.tif deprecated/frame000.tif
```

- [ ] **Step 3: Commit**

```bash
git add C++/deprecated/
git commit -m "refactor: move dead/unused files to C++/deprecated/

Moved 9 files that are not part of the active build:
- Bacilli.hpp/.cpp (dead, never compiled)
- Sphere.hpp/.cpp (compiled but never instantiated)
- Cell.hpp (vestigial abstract base, nothing inherits it)
- mathhelper.hpp (orphaned, included by nothing)
- args.cpp (not compiled, uses undefined cxxopts)
- pseudo-code.txt (outdated)
- frame000.tif (corrupted stray test file)"
```

---

## Task 2: Remove dead Cell.hpp includes and fix dead method signatures

**Context from code verification:**
- Spheroid does NOT inherit Cell — class declaration is already `class Spheroid {` (Spheroid.hpp:71)
- `CellParams` is defined in ConfigTypes.hpp:246 (NOT in Cell.hpp) — SpheroidParams inheritance is safe
- Cell.hpp is included by 4 active files but provides nothing they actually use
- Frame.hpp has two private methods (`costOfPerturb`, `getSynthPerturbedCells`) that take `const Cell &oldCell` but **never use the parameter** in their implementations (Frame.cpp:440-488)
- Frame.hpp declares `gradientDescent()` but it's **entirely commented out** in Frame.cpp:491-571

**Files:**

- Modify: `C++/includes/Spheroid.hpp:17` — remove `#include "Cell.hpp"`
- Modify: `C++/includes/Frame.hpp:5,13,58-59` — remove Cell.hpp + Sphere.hpp includes, fix method signatures, remove gradientDescent decl
- Modify: `C++/includes/CellFactory.hpp:14` — remove `#include "Cell.hpp"`
- Modify: `C++/includes/Lineage.hpp:5` — remove `#include "Cell.hpp"`
- Modify: `C++/src/Frame.cpp:440,465,491-571` — fix method signatures, remove dead commented code

- [ ] **Step 1: Update Spheroid.hpp — remove dead include**

Spheroid.hpp line 17, remove:

```cpp
#include "Cell.hpp"
```

Spheroid.hpp already includes `<opencv2/opencv.hpp>` and the standard headers it needs. `CellParams` (inherited by SpheroidParams) comes from ConfigTypes.hpp, which Spheroid.hpp gets transitively through... actually it doesn't include ConfigTypes.hpp directly. We need to add it.

Replace line 17:

```cpp
#include "Cell.hpp"
```

With:

```cpp
#include "ConfigTypes.hpp"
```

This gives Spheroid.hpp direct access to `CellParams`, `SpheroidConfig`, `SimulationConfig`, and `PerturbParams` — all of which it uses.

- [ ] **Step 2: Update Frame.hpp — remove dead includes and fix signatures**

Frame.hpp line 5, remove:

```cpp
#include "Cell.hpp"
```

Frame.hpp line 13, remove:

```cpp
#include "Sphere.hpp"
```

Frame.hpp already includes `ConfigTypes.hpp`, `types.hpp`, and `Spheroid.hpp` directly, so nothing is lost.

Frame.hpp lines 58-59, change from:

```cpp
    Cost costOfPerturb(const std::string &perturbParam, float perturbVal, size_t index, const Cell &oldCell);
    ParamImageMap getSynthPerturbedCells(size_t index, const ParamValMap &params, float perturbLength, const Cell &oldCell);
```

To (remove unused `const Cell &oldCell` parameter):

```cpp
    Cost costOfPerturb(const std::string &perturbParam, float perturbVal, size_t index);
    ParamImageMap getSynthPerturbedCells(size_t index, const ParamValMap &params, float perturbLength);
```

The `oldCell` parameter is never read in either implementation — both functions operate on `cells[index]` directly.

Frame.hpp line 41, remove the dead declaration:

```cpp
    Cost gradientDescent();
```

- [ ] **Step 3: Update Frame.cpp — match new signatures, remove dead code**

Frame.cpp line 440, change from:

```cpp
Cost Frame::costOfPerturb(const std::string &perturbParam, float perturbVal, size_t index, const Cell &oldCell)
```

To:

```cpp
Cost Frame::costOfPerturb(const std::string &perturbParam, float perturbVal, size_t index)
```

Frame.cpp lines 460-465, change from:

```cpp
ParamImageMap
Frame::getSynthPerturbedCells(
    size_t index,
    const ParamValMap &params,
    float perturbLength,
    const Cell &oldCell)
```

To:

```cpp
ParamImageMap
Frame::getSynthPerturbedCells(
    size_t index,
    const ParamValMap &params,
    float perturbLength)
```

Frame.cpp lines 491-571 — delete the entire commented-out `gradientDescent()` block (80 lines of dead commented code that references `Cell*` pointers).

- [ ] **Step 4: Update CellFactory.hpp — remove dead include**

CellFactory.hpp line 14, remove:

```cpp
#include "Cell.hpp"
```

CellFactory.hpp already includes `ConfigTypes.hpp` and `Spheroid.hpp` directly.

- [ ] **Step 5: Update Lineage.hpp — remove dead include**

Lineage.hpp line 5, remove:

```cpp
#include "Cell.hpp"
```

Lineage.hpp already includes `Frame.hpp`, `ConfigTypes.hpp`, and `Spheroid.hpp` directly.

- [ ] **Step 6: Verify no remaining Cell.hpp references in active code**

Search for any remaining references:

```bash
grep -r "Cell.hpp\|Cell &\|Cell\*\|Cell \*" C++/includes/ C++/src/ --include="*.hpp" --include="*.cpp"
```

After this task, the only references to `Cell` should be `CellParams` and `CellConfig` (both defined in ConfigTypes.hpp, both actively used).

- [ ] **Step 7: Commit**

```bash
git add C++/includes/Spheroid.hpp C++/includes/Frame.hpp C++/includes/CellFactory.hpp C++/includes/Lineage.hpp C++/src/Frame.cpp
git commit -m "refactor: remove all Cell.hpp includes and dead Cell& method signatures

Cell.hpp was vestigial — Spheroid does not inherit Cell (already
standalone class), and CellParams is defined in ConfigTypes.hpp.
Removed #include Cell.hpp from Spheroid.hpp, Frame.hpp,
CellFactory.hpp, Lineage.hpp. Removed #include Sphere.hpp from
Frame.hpp. Removed unused const Cell& parameter from
costOfPerturb() and getSynthPerturbedCells() in Frame.hpp/.cpp.
Removed dead commented-out gradientDescent() (80 lines)."
```

---

## Task 3: Flatten src/Cells/ directory

Only Spheroid.cpp remains in `src/Cells/` after Task 1. Move it up to `src/` and update its include path.

**Files:**

- Move: `C++/src/Cells/Spheroid.cpp` -> `C++/src/Spheroid.cpp`
- Modify: `C++/src/Spheroid.cpp:1` — update include path
- Modify: `C++/CMakeLists.txt` — update source file path, remove include dir

- [ ] **Step 1: Move Spheroid.cpp**

```bash
cd C++
git mv src/Cells/Spheroid.cpp src/Spheroid.cpp
rmdir src/Cells  # Should be empty now
```

- [ ] **Step 2: Update include path in Spheroid.cpp**

Line 1, change from:

```cpp
#include "../../includes/Spheroid.hpp"
```

To:

```cpp
#include "../includes/Spheroid.hpp"
```

The path changes from `src/Cells/` (2 levels deep) to `src/` (1 level deep).

- [ ] **Step 3: Update CMakeLists.txt source list**

Current source list in `add_executable`:

```cmake
add_executable(celluniverse
    src/main.cpp
    src/CellFactory.cpp
    src/Lineage.cpp
    src/Frame.cpp
    src/LineageViewer.cpp
    src/Cells/Sphere.cpp
    src/Cells/Spheroid.cpp
)
```

Change to:

```cmake
add_executable(celluniverse
    src/main.cpp
    src/CellFactory.cpp
    src/Lineage.cpp
    src/Frame.cpp
    src/LineageViewer.cpp
    src/Spheroid.cpp
)
```

Removes `src/Cells/Sphere.cpp` (now in deprecated/) and updates `src/Cells/Spheroid.cpp` -> `src/Spheroid.cpp`.

Also remove `src/Cells` from include directories if present:

```cmake
# Remove this line if it exists:
# ${cwd}/src/Cells
```

- [ ] **Step 4: Commit**

```bash
git add C++/src/Spheroid.cpp C++/CMakeLists.txt
git commit -m "refactor: flatten src/Cells/ — move Spheroid.cpp to src/

Only Spheroid remains as the active cell type. Moved to src/ and
updated include path. Removed Sphere.cpp from CMakeLists.txt (now
in deprecated/). Removed src/Cells/ directory."
```

---

## Task 4: Reorganize examples/ into config/, scripts/, data/

The current `examples/` directory mixes configuration files, shell scripts, data files, and output directories. Split into logical subdirectories.

**Files:**

- Create: `C++/config/`, `C++/scripts/`, `C++/data/`
- Move: 6 config files, 6 scripts, 1 data directory, 1 doc file
- Modify: `C++/scripts/run_celluniverse.sh` and other scripts — update relative paths

- [ ] **Step 1: Create new directories**

```bash
cd C++
mkdir -p config scripts data
```

- [ ] **Step 2: Move configuration files**

```bash
git mv examples/config.yaml config/config.yaml
git mv examples/initial.csv config/initial.csv
git mv examples/initial_auto.csv config/initial_auto.csv
git mv examples/initial_embryo.csv config/initial_embryo.csv
git mv examples/user_input_configurations.ini config/user_input_configurations.ini
git mv examples/runauto.args config/runauto.args
```

- [ ] **Step 3: Move shell scripts and utilities**

```bash
git mv examples/run_celluniverse.sh scripts/run_celluniverse.sh
git mv examples/run_original.sh scripts/run_original.sh
git mv examples/run_embryo.sh scripts/run_embryo.sh
git mv examples/runauto.sh scripts/runauto.sh
git mv examples/remove_old_outputs.sh scripts/remove_old_outputs.sh
git mv examples/convert_png_to_tiff.py scripts/convert_png_to_tiff.py
```

- [ ] **Step 4: Move input data**

```bash
git mv examples/input data/input
```

This moves the entire `input/original_data/` tree with all 41 TIFF files.

- [ ] **Step 5: Move details.md to docs/**

```bash
git mv examples/details.md docs/details.md
```

`details.md` is the comprehensive technical specification — it belongs with the other docs, not in examples.

- [ ] **Step 6: Update run_celluniverse.sh paths**

The main run script references relative paths to config, data, and build directories. Read the script first, then update all relative paths.

Key path changes inside `run_celluniverse.sh`:

| Old (relative to examples/) | New (relative to scripts/) |
|-----|-----|
| `config.yaml` | `../config/config.yaml` |
| `initial.csv` | `../config/initial.csv` |
| `user_input_configurations.ini` | `../config/user_input_configurations.ini` |
| `runauto.args` | `../config/runauto.args` |
| `input/` or `input/original_data/` | `../data/input/` or `../data/input/original_data/` |
| `../build` | `../build` (same — scripts/ is same depth as examples/) |

**Note:** The script may reference paths from its INI file (`user_input_configurations.ini`). The INI file itself contains `input_path`, `cell_config_file`, `initial_csv_file` — these need updating to match new relative paths from the script's working directory.

Similarly update `run_original.sh`, `run_embryo.sh`, `runauto.sh` if they reference moved files.

Also update `remove_old_outputs.sh` to look for output directories in the correct location.

- [ ] **Step 7: Update user_input_configurations.ini paths**

The INI file at `config/user_input_configurations.ini` has paths like:

```ini
input_path=input/original_data
cell_config_file=config.yaml
initial_csv_file=initial.csv
```

These were relative to `examples/`. They need to be updated to be relative to wherever the script's working directory will be. Read the run script to understand how it resolves these paths, then update accordingly.

- [ ] **Step 8: Handle remaining examples/ directory**

After moving everything, `examples/` should contain only untracked `output_*/` directories (created by runs, not in git).

If empty of tracked files, the directory can be removed from git. Output directories will be recreated by future runs — update scripts to write output to a new location (e.g., `C++/output/` or `C++/data/output/`).

Leave untracked output directories for user to clean up manually.

- [ ] **Step 9: Update .gitignore**

Add patterns for new output locations:

```gitignore
# Outputs (new location)
C++/output_*/
C++/data/output_*/
```

- [ ] **Step 10: Commit**

```bash
git add C++/config/ C++/scripts/ C++/data/ C++/docs/details.md C++/.gitignore
git commit -m "refactor: reorganize examples/ into config/, scripts/, data/

- config/ — YAML configs, initial CSVs, INI presets, args files
- scripts/ — shell run scripts and utilities
- data/input/ — TIFF image stacks
- docs/details.md — moved from examples/ (technical documentation)

Updated relative paths in run scripts and INI file to match new layout."
```

---

## Task 5: Clean up CMakeLists.txt include paths

After Tasks 1-4, some include directories in CMakeLists.txt are stale.

**Files:**

- Modify: `C++/CMakeLists.txt`

- [ ] **Step 1: Review and clean include_directories**

Current include directories likely:

```cmake
include_directories(
    ${cwd}/includes
    ${cwd}/src
    ${cwd}/src/Cells
    ${cwd}/lib/yaml-cpp/include
)
```

Change to:

```cmake
include_directories(
    ${cwd}/includes
    ${cwd}/src
    ${cwd}/lib/yaml-cpp/include
)
```

Remove `${cwd}/src/Cells` — the directory no longer exists.

- [ ] **Step 2: Verify no other stale references**

Search CMakeLists.txt for any references to:

- `Cells/` — should be gone
- `Sphere` — should be gone
- `examples/` — should be updated if referenced
- `args.cpp` — should not be present (was never compiled)

- [ ] **Step 3: Commit**

```bash
git add C++/CMakeLists.txt
git commit -m "refactor: clean up CMakeLists.txt include paths

Removed stale src/Cells include directory."
```

---

## Task 6: Update documentation references

After restructuring, documentation that references old file paths needs updating.

**Files:**

- Modify: `C++/docs/details.md` — update file paths throughout
- Modify: `C++/README.md` — update build/run instructions
- Modify: `.claude/CLAUDE.md` — update file reference table and commands
- Modify: `.claude/rules/codebase.md` — update class hierarchy and file roles

- [ ] **Step 1: Update docs/details.md**

Search and replace old paths throughout the document:

| Old Path | New Path |
|----------|----------|
| `src/Cells/Spheroid.cpp` | `src/Spheroid.cpp` |
| `examples/config.yaml` | `config/config.yaml` |
| `examples/initial.csv` | `config/initial.csv` |
| `examples/details.md` | `docs/details.md` |

Remove references to `Sphere.cpp`, `Bacilli.cpp`, `Cell.hpp` as active files. Update the file reference table (Section 11) at the end.

- [ ] **Step 2: Update C++/README.md**

Update run instructions:

```markdown
# Run (from C++/scripts/)
./run_celluniverse.sh ../config/user_input_configurations.ini default

# Direct execution (from C++/)
./build/celluniverse 1 19 data/input/original_data output/ config/config.yaml config/initial.csv
```

- [ ] **Step 3: Update .claude/CLAUDE.md**

Update:

- File reference table — remove Sphere, Bacilli, Cell.hpp rows; update Spheroid.cpp path to `src/Spheroid.cpp`; add note about `deprecated/` directory
- Commands section — new script/config paths
- Architecture section — note that Spheroid is standalone (no Cell base class)

- [ ] **Step 4: Update .claude/rules/codebase.md**

Update:

- Class hierarchy — remove Sphere, Bacilli, Cell abstract base; show Spheroid as standalone
- File roles — update all paths, remove deprecated files from active list
- Add note about `deprecated/` directory and its contents

- [ ] **Step 5: Commit**

```bash
git add C++/docs/details.md C++/README.md .claude/CLAUDE.md .claude/rules/codebase.md
git commit -m "docs: update file path references after restructuring

Updated details.md, README.md, CLAUDE.md, and codebase.md to
reflect new directory layout: src/Spheroid.cpp, config/, scripts/,
data/, deprecated/. Removed references to deprecated files.
Corrected class hierarchy: Spheroid is standalone, no Cell base."
```

---

## Task 7: Write changelog

**Files:**

- Create: `C++/docs/changelog_restructure_2026-03-27.md`

- [ ] **Step 1: Write changelog following project format**

Document every file moved, every include changed, every path updated with exact before/after code. Follow the format in `changelogv3_after_brightness_fix.md`.

Key sections:

1. Files moved to `deprecated/` (9 files, with original locations)
2. Cell.hpp include removal from Spheroid.hpp, Frame.hpp, CellFactory.hpp, Lineage.hpp (before/after includes)
3. Sphere.hpp include removal from Frame.hpp
4. Frame.hpp/Frame.cpp: removed unused `const Cell &oldCell` parameter from `costOfPerturb()` and `getSynthPerturbedCells()` (before/after signatures)
5. Frame.hpp/Frame.cpp: removed dead `gradientDescent()` declaration and 80-line commented-out implementation
6. Spheroid.hpp: replaced `#include "Cell.hpp"` with `#include "ConfigTypes.hpp"` (before/after)
7. Spheroid.cpp moved from `src/Cells/` to `src/` and include path updated (before/after)
8. CMakeLists.txt changes: source list and include directories (before/after)
9. `examples/` reorganized into `config/`, `scripts/`, `data/` (complete file mapping)
10. Run script and INI path updates (before/after)
11. Documentation path updates

- [ ] **Step 2: Copy changelog to memory directory**

```bash
cp C++/docs/changelog_restructure_2026-03-27.md \
   ~/.claude/projects/-Users-jihangli-MCS-3D-Cell-Tracking-CellUniverse/memory/
```

- [ ] **Step 3: Final commit**

```bash
git add C++/docs/changelog_restructure_2026-03-27.md
git commit -m "docs: add changelog for C++ directory restructuring"
```

---

## Task 8: Remove dead code from Spheroid

**Files:**
- Modify: `C++/includes/Spheroid.hpp`
- Modify: `C++/src/Spheroid.cpp`

- [ ] **Step 1: Remove dead member variables from Spheroid class**

Remove from `Spheroid.hpp` private section:

```cpp
// REMOVE — always false, never set to true
bool dormant;

// REMOVE — always 0, never read; rotation uses _theta_x/y/z
double _rotation;

// REMOVE — never populated in main constructor; segfaults if accessed
std::vector<double> _x_vec;
std::vector<double> _y_vec;
std::vector<double> _z_vec;
```

Update both constructors to remove `_rotation(0)` and `dormant(false)` from initializer lists.

- [ ] **Step 2: Remove dead static member**

Remove from `Spheroid.hpp`:

```cpp
static SpheroidParams paramClass;  // never referenced anywhere
```

Remove from `Spheroid.cpp`:

```cpp
SpheroidParams Spheroid::paramClass = SpheroidParams();
```

- [ ] **Step 3: Remove dead methods**

Remove declarations from `Spheroid.hpp`:

```cpp
float major_magnitude() const;   // never called; crashes on _x_vec access
float minor_magnitude() const;   // never called; crashes on _z_vec access
int get_matrix_size();            // never called
std::vector<double> getShapeAt(double z) const;  // never called
bool checkIfCellsValid(const std::vector<Spheroid> &spheroids);  // only in dead perturb()
static bool checkIfCellsOverlap(const std::vector<Spheroid> &spheroids);  // only via checkIfCellsValid
```

Remove implementations from `Spheroid.cpp`:

```cpp
float Spheroid::major_magnitude() const { ... }     // ~3 lines
float Spheroid::minor_magnitude() const { ... }     // ~3 lines
int Spheroid::get_matrix_size() { ... }              // ~3 lines
std::vector<double> Spheroid::getShapeAt(double z) const { ... }  // ~5 lines
bool Spheroid::checkIfCellsOverlap(const std::vector<Spheroid> &spheroids) { ... }  // ~25 lines
```

Also remove the `_get_magnitude()` free function if no longer needed.

- [ ] **Step 4: Remove dormant checks from draw/drawOutline**

In `Spheroid.cpp` `draw()`, remove:

```cpp
if (dormant) { return; }
```

In `Spheroid.cpp` `drawOutline()`, remove:

```cpp
if (dormant) return;
```

- [ ] **Step 5: Remove unused `cellMap` parameter from draw()**

In `Spheroid.hpp`, change:

```cpp
void draw(cv::Mat &image, SimulationConfig simulationConfig, cv::Mat *cellMap = nullptr, float z = 0) const;
```

To:

```cpp
void draw(cv::Mat &image, const SimulationConfig &simulationConfig, float z = 0) const;
```

Update all call sites in `Frame.cpp` (`generateSynthFrame`, `generateSynthFrameFast`):

```cpp
// Before:
cell.draw(synthImage, simulationConfig, nullptr, z);
// After:
cell.draw(synthImage, simulationConfig, z);
```

Update implementation in `Spheroid.cpp` to remove `(void)cellMap;` line and match new signature.

- [ ] **Step 6: Remove dead SpheroidParams vector constructor and parseParams**

In `Spheroid.hpp`, the `SpheroidParams` constructor taking `_x_vec, _y_vec, _z_vec` (lines 43-53) and `parseParams()` (lines 55-65) are never used since cells are always created with scalar `(x, y, z, majorR, minorR)`. Remove both.

- [ ] **Step 7: Commit**

```bash
git commit -m "refactor: remove dead code from Spheroid

Removed: dormant (always false), _rotation (unused), _x_vec/_y_vec/_z_vec
(never populated), paramClass (never referenced), major/minor_magnitude
(crash risk), get_matrix_size, getShapeAt, checkIfCellsValid/Overlap
(only in dead perturb()), cellMap param from draw(), vector-based
SpheroidParams constructor and parseParams."
```

---

## Task 9: Remove dead code from Frame

**Files:**
- Modify: `C++/includes/Frame.hpp`
- Modify: `C++/src/Frame.cpp`

- [ ] **Step 1: Remove dead `perturb()` method**

Remove from `Frame.hpp`:

```cpp
CostCallbackPair perturb();
```

Remove from `Frame.cpp` the entire `Frame::perturb()` implementation (~48 lines). This was superseded by `perturbCell()` and is never called.

- [ ] **Step 2: Remove dead `padRealFrame()` method**

Remove from `Frame.hpp`:

```cpp
void padRealFrame();
```

Remove from `Frame.cpp` the entire `Frame::padRealFrame()` implementation (~14 lines) and the commented-out call in the constructor:

```cpp
// TODO: Fix padding
//    padRealImage();
```

- [ ] **Step 3: Remove dead `costOfPerturb()` and `getSynthPerturbedCells()`**

Remove from `Frame.hpp`:

```cpp
Cost costOfPerturb(const std::string &perturbParam, float perturbVal, size_t index);
ParamImageMap getSynthPerturbedCells(size_t index, const ParamValMap &params, float perturbLength);
```

Remove from `Frame.cpp` both implementations (~48 lines total). These are remnants of old gradient descent optimization, never called.

- [ ] **Step 4: Remove dead `split()` method (if present)**

`Frame::split()` just wraps `trySplitCell()` with a random index and is never called from the optimize loop (which uses `randomSplitCell()`). Remove if still present.

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor: remove dead methods from Frame

Removed: perturb() (superseded by perturbCell()), padRealFrame()
(disabled in constructor), costOfPerturb(), getSynthPerturbedCells()
(old gradient descent remnants), split() (never called)."
```

---

## Task 10: Remove dead code from ConfigTypes, CellFactory, main

**Files:**
- Modify: `C++/includes/ConfigTypes.hpp`
- Modify: `C++/src/CellFactory.cpp`
- Modify: `C++/src/main.cpp`
- Modify: `C++/includes/types.hpp`

- [ ] **Step 1: Remove unused `z_values` from SimulationConfig**

In `ConfigTypes.hpp`, remove:

```cpp
std::vector<int> z_values;
```

In `main.cpp`, remove:

```cpp
config.simulation.z_values.clear();
for (int j = 0; j < slices; ++j)
{
    config.simulation.z_values.push_back(j - slices / 2);
}
```

- [ ] **Step 2: Fix CellFactory sphere branch**

In `CellFactory.cpp`, replace:

```cpp
if (cellType == "sphere") {
    Spheroid::cellConfig = *config.cell;
} if (cellType == "spheroid") {
    Spheroid::cellConfig = *config.cell;
}
else {
    throw std::invalid_argument("Invalid cell type: " + config.cellType);
}
```

With:

```cpp
if (cellType == "spheroid") {
    Spheroid::cellConfig = *config.cell;
} else {
    throw std::invalid_argument("Invalid cell type: " + config.cellType);
}
```

Removes the dead "sphere" branch and fixes the `if`/`else if` logic error.

- [ ] **Step 3: Remove dead LineageViewer usage and wait loop from main.cpp**

Remove the viewer creation and dead wait loop:

```cpp
// REMOVE — viewer.update() is commented out, window is never created
LineageViewer viewer;

// REMOVE — the CellViz building block (it feeds the disabled viewer)
std::vector<LineageViewer::CellViz> viz;
const auto &cellsNow = lineage.getCells(frame);
viz.reserve(cellsNow.size());
for (const auto &cell : cellsNow) { ... }
// viewer.update(args.firstFrame + frame, viz);

// REMOVE — waits for a window that was never created (hangs forever)
while (true)
{
    int key = cv::waitKey(30);
    if (cv::getWindowProperty("Cell Lineage (Realtime)", cv::WND_PROP_VISIBLE) < 1)
    {
        break;
    }
}
```

Also remove `#include "LineageViewer.hpp"` from main.cpp.

**Note:** Keep `LineageViewer.hpp` and `LineageViewer.cpp` in the codebase — they are incomplete but potentially useful for future integration (see Task 11). Just don't create or call the viewer in main.

- [ ] **Step 4: Remove commented-out include from types.hpp**

Remove:

```cpp
//#include "Sphere.hpp"
```

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor: remove dead code from ConfigTypes, CellFactory, main

Removed: z_values (populated but never read), dead sphere branch in
CellFactory (fixed if/else logic), disabled LineageViewer usage and
infinite wait loop in main, commented Sphere.hpp include in types.hpp."
```

---

## Task 11: Reconstruct — items for future work

These items are not dead code but are incomplete or need redesign:

### 11a. Cell lineage tracking (primary open feature)

**Current:** Cell naming uses string concatenation (`name + "0"`, `name + "1"`) for parent-child relationships. `cells.csv` records per-frame state but does not explicitly encode parent-child relationships.

**Needed:** Proper lineage data structure with explicit parent pointers, split frame tracking, and lineage tree output. This is the primary incomplete feature mentioned in CLAUDE.md.

**Files affected:** `CellUniverse.cpp`, `Spheroid.hpp` (SpheroidParams needs parent field), output CSV format.

### 11b. LineageViewer integration

**Current:** `LineageViewer.hpp/.cpp` exist with a partially implemented 2D visualization. The `update()` call is commented out in main.cpp.

**Options:**
1. **Fully integrate** — uncomment `viewer.update()`, fix any issues, add proper window lifecycle
2. **Remove entirely** — delete LineageViewer.hpp/.cpp from build if no plans to use it

**Recommendation:** Keep the files but don't call from main until lineage tracking (11a) is implemented. The viewer depends on having meaningful lineage data.

### 11c. SimulationConfig `cell_color` field

**Current:** `cell_color: 0.6` is parsed and stored but never used — rendering now uses per-cell `_brightness`. Consider removing from config and SimulationConfig to avoid confusion.

---

## Summary of All Changes

| # | What | Files Affected | Risk |
|---|------|---------------|------|
| 1 | Move 9 dead files to deprecated/ | 9 file moves | None — files not compiled or referenced |
| 2 | Remove Cell.hpp includes + fix Frame dead signatures | Spheroid.hpp, Frame.hpp, Frame.cpp, CellFactory.hpp, Lineage.hpp | **Medium** — removes includes and changes 2 method signatures; needs rebuild to verify |
| 3 | Flatten src/Cells/ | Spheroid.cpp, CMakeLists.txt | **Medium** — changes include path and build file; needs rebuild |
| 4 | Reorganize examples/ | 13 file moves, script + INI path updates | **Medium** — run scripts need path updates; user must verify scripts work |
| 5 | Clean CMakeLists.txt | CMakeLists.txt | Low — removing stale include path |
| 6 | Update documentation | 4 doc files | None — documentation only |
| 7 | Write changelog | 1 new file | None — documentation only |
| 8 | Remove dead code from Spheroid | Spheroid.hpp, Spheroid.cpp | **Medium** — removes 6 member vars, 6 methods, changes draw() signature; needs rebuild |
| 9 | Remove dead code from Frame | Frame.hpp, Frame.cpp | Low — removes 4 methods never called |
| 10 | Remove dead code from ConfigTypes, CellFactory, main | ConfigTypes.hpp, CellFactory.cpp, main.cpp, types.hpp | **Medium** — removes dead wait loop, fixes CellFactory logic; needs rebuild |
| 11 | Reconstruct items (future) | — | — planning only, no code changes |

**Total: 10 implementation tasks + 1 future planning task, ~45 steps**

Tasks 1-7 were the original restructure plan.
Tasks 8-10 are new dead code removal (added 2026-03-27 evening).
Task 11 documents reconstruction items for future sessions.

**Build verification required after Tasks 2, 3, 5, 8, 9, and 10.** User must run `cmake -S .. -B . && cmake --build .` from `C++/build/` to confirm the project compiles.

**Run verification required after Task 4.** User must test that `scripts/run_celluniverse.sh` correctly finds config files, data, and build output with updated paths.

---

## Code Verification Notes (2026-03-27)

These findings were verified by reading actual source code, not relying on the UML diagram:

1. **Spheroid does NOT inherit Cell** — `class Spheroid {` at Spheroid.hpp:71 (no `: public Cell`)
2. **Sphere does NOT inherit Cell** — `class Sphere {` at Sphere.hpp:33
3. **Bacilli DOES inherit Cell** — `class Bacilli : public Cell` at Bacilli.hpp:6 (but Bacilli is dead code)
4. **CellParams lives in ConfigTypes.hpp:246** — NOT in Cell.hpp. Safe to remove Cell.hpp.
5. **Frame::costOfPerturb** takes `const Cell &oldCell` (Frame.hpp:58) but **never uses it** (Frame.cpp:440-457)
6. **Frame::getSynthPerturbedCells** takes `const Cell &oldCell` (Frame.hpp:59) but **never uses it** (Frame.cpp:460-488)
7. **Frame::gradientDescent** is declared (Frame.hpp:41) but **entirely commented out** in Frame.cpp:491-571
8. **Cell.hpp is included by**: Spheroid.hpp:17, Frame.hpp:5, CellFactory.hpp:14, Lineage.hpp:5 — all unnecessary
9. **`dormant`** — always `false` in Spheroid, checked in draw()/drawOutline() but never set to `true`
10. **`_rotation`** — always `0`, never read; actual rotation uses `_theta_x/y/z`
11. **`_x_vec/_y_vec/_z_vec`** — empty in main constructor; `major_magnitude()`/`minor_magnitude()` would crash
12. **`perturb()`** — superseded by `perturbCell()`, never called from optimize loop
13. **`padRealFrame()`** — disabled in Frame constructor, never called
14. **`z_values`** — populated in main.cpp but never read anywhere
15. **LineageViewer** — created in main.cpp but `update()` is commented out; wait loop hangs forever
16. **CellFactory** — `if (sphere)` branch is dead; `if`/`else if` logic error exists
