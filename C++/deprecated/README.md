# Deprecated Files

Files in this directory were removed from the active build on 2026-03-27 during
the C++ directory restructuring. They are preserved for historical reference.

## Contents

| File | Original Location | Reason |
|------|-------------------|--------|
| Bacilli.hpp | includes/ | Dead code -- never compiled, incomplete implementation |
| Bacilli.cpp | src/Cells/ | Dead code -- not in CMakeLists.txt, references undefined Rectangle class |
| Sphere.hpp | includes/ | Compiled but never instantiated in execution path |
| Sphere.cpp | src/Cells/ | Compiled but never called -- CellFactory only creates Spheroid |
| Cell.hpp | includes/ | Vestigial abstract base -- no class inherits it on this branch, no polymorphic usage |
| mathhelper.hpp | includes/ | Orphaned -- not included by any file in the project |
| args.cpp | C++/ root | Not compiled -- uses undefined cxxopts library |
| pseudo-code.txt | C++/ root | Outdated pseudocode, does not reflect current algorithm |
| frame000.tif | C++/ root | Stray test file with corrupted dimensions (0x0) |

## Restoring Files

If any of these files are needed again, copy them back to their original location
and update CMakeLists.txt accordingly.
