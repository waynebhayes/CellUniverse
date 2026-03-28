# Changelog: Project Directory Restructuring (2026-03-27)

## Summary

Reorganized the C++ project directory structure to separate concerns: deprecated dead code moved out of active source tree, config/scripts/data separated from examples/, Spheroid.cpp moved to top-level src/.

---

## 1. Files moved to `C++/deprecated/`

Nine files moved from active source/include directories to `C++/deprecated/`:

| File | Original location | Content |
|------|-------------------|---------|
| `Cell.hpp` | `C++/includes/Cell.hpp` | Abstract base class (unused since Spheroid became standalone) |
| `Sphere.hpp` | `C++/includes/Sphere.hpp` | Sphere subclass of Cell (unused) |
| `Sphere.cpp` | `C++/src/Sphere.cpp` | Sphere implementation (unused) |
| `Bacilli.hpp` | `C++/includes/Bacilli.hpp` | Bacilli subclass of Cell (unused) |
| `Bacilli.cpp` | `C++/src/Bacilli.cpp` | Bacilli implementation (unused) |
| `mathhelper.hpp` | `C++/includes/mathhelper.hpp` | Math utilities (unused) |
| `args.cpp` | `C++/src/args.cpp` | Argument parsing (replaced by direct argv parsing in main.cpp) |
| `pseudo-code.txt` | `C++/src/pseudo-code.txt` | Design notes |
| `README.md` | `C++/deprecated/README.md` | Old readme preserved |

Also already present in deprecated: `frame000.tif` (test image).

## 2. `#include "Cell.hpp"` removed from 4 files

Removed the dead include from files that no longer reference the Cell base class:

- `C++/includes/Spheroid.hpp`
- `C++/includes/Frame.hpp`
- `C++/src/CellFactory.cpp`
- `C++/src/Frame.cpp`

## 3. `#include "Sphere.hpp"` removed from `C++/includes/Frame.hpp`

Dead include for the unused Sphere class.

## 4. Frame method signatures fixed

Removed `const Cell& oldCell` parameter from methods in `C++/includes/Frame.hpp` and `C++/src/Frame.cpp` that referenced the deleted Cell base class:

- `generateSynthFrameFast()` -- parameter changed from `const Cell& oldCell` to `const Spheroid& oldCell`
- Other methods updated to use `Spheroid` directly instead of `Cell` references

## 5. `gradientDescent()` removed

Removed unused `gradientDescent()` method from:
- `C++/includes/Frame.hpp` (declaration)
- `C++/src/Frame.cpp` (implementation)

This method was dead code -- all optimization uses Monte Carlo perturbation.

## 6. `Spheroid.cpp` moved from `src/Cells/` to `src/`

- **From:** `C++/src/Cells/Spheroid.cpp`
- **To:** `C++/src/Spheroid.cpp`
- Include path updated in the file: `#include "Spheroid.hpp"` (path unchanged since includes/ is a flat directory)
- `C++/src/Cells/` directory removed (was the only file in it)

## 7. `CMakeLists.txt` updated

Source file paths updated to reflect:
- `src/Spheroid.cpp` (was `src/Cells/Spheroid.cpp`)
- Removed references to `src/Sphere.cpp`, `src/args.cpp`, `src/Bacilli.cpp`

## 8. `C++/examples/` reorganized into `config/`, `scripts/`, `data/`

The monolithic `examples/` directory was split into purpose-specific directories:

| New location | Files moved | Original location |
|-------------|-------------|-------------------|
| `C++/config/` | `config.yaml`, `initial.csv`, `initial_auto.csv`, `initial_embryo.csv`, `user_input_configurations.ini`, `runauto.args` | `C++/examples/` |
| `C++/scripts/` | `run_celluniverse.sh`, `run_original.sh`, `run_embryo.sh`, `runauto.sh`, `convert_png_to_tiff.py`, `remove_old_outputs.sh` | `C++/examples/` |
| `C++/data/input/` | `original_data/` (input images) | `C++/examples/` |

## 9. INI file paths updated

`C++/config/user_input_configurations.ini` -- all paths updated to reflect the new directory structure:

- Config file paths: `examples/config.yaml` -> `config/config.yaml`
- Initial CSV paths: `examples/initial.csv` -> `config/initial.csv`
- Input data paths: `examples/original_data` -> `data/input/original_data`
- Output paths: updated to `outputs/` or appropriate locations

## 10. Documentation updated

- **`C++/docs/details.md`**: `src/Cells/Spheroid.cpp` -> `src/Spheroid.cpp` (2 occurrences)
- **`C++/README.md`**: Complete rewrite with new directory structure, updated build/run commands
- **`.claude/CLAUDE.md`**: `C++/examples/details.md` -> `C++/docs/details.md` (reference fix)
- **`.claude/rules/codebase.md`**: Removed "(moved from src/Cells/)" parenthetical from Spheroid.cpp entry
- **`.claude/rules/changelog.md`**: `C++/examples/details.md` -> `C++/docs/details.md`
- **MEMORY.md**: `C++/examples/config.yaml` -> `C++/config/config.yaml`
