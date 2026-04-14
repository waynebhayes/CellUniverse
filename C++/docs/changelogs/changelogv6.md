# Changelog v6

Opened 2026-04-13 (late evening). Covers the iterative PCA shape-fit work after v5 closed.

---

## 2026-04-13 (late evening patch 2): Fix PCA oscillation — rank assignment

**Status:** ACTIVE

**Problem:** On the first 3-frame run after the iterative PCA rewrite, `[PCA Shape]` logs showed period-3 oscillation through all 15 iters:

```
e3d03 iter 2: R=(19.15, 16.25, 24.78)
e3d03 iter 3: R=(23.58, 19.20, 16.42)
e3d03 iter 4: R=(19.42, 16.61, 25.75)
```

Symptoms: `dR` never dropped below the convergence threshold; `axisAng` stayed 35-50° iter after iter; eigenvalues cycled through slot labels.

**Root cause:** greedy axis-matching by `|current_axis · pca_axis|` chose the largest-absolute-dot PCA axis for each slot in slot order (a → b → c). For cells where the current orientation was close to 45° off the true axes, small perturbations in the mask changed dot-product magnitudes just enough to reassign pca axis ids between slots each iteration. Radii followed the reassignment, so slot a could hold the largest eigenvalue one iter and the smallest the next.

**Fix:** `C++/src/Frame.cpp` (inside `Frame::calibrateCellShapeViaPca`) — replaced greedy matching with strict eigenvalue-rank assignment.

**Before:**
```cpp
// Greedy-match PCA axes to current a/b/c by largest |dot|.
int pcaForCur[3] = {-1, -1, -1};
bool pcaUsed[3] = {false, false, false};
for (int ci = 0; ci < 3; ++ci) {
    int bestIdx = -1;
    double bestAbs = -1.0;
    for (int pj = 0; pj < 3; ++pj) {
        if (pcaUsed[pj]) continue;
        const double d = curAxis[ci].x * pcaAxis[pj].x +
                         curAxis[ci].y * pcaAxis[pj].y +
                         curAxis[ci].z * pcaAxis[pj].z;
        if (std::abs(d) > bestAbs) { bestAbs = std::abs(d); bestIdx = pj; }
    }
    pcaForCur[ci] = bestIdx;
    pcaUsed[bestIdx] = true;
}

cv::Point3f matchedAxis[3];
double matchedVariance[3];
double maxAxisAngle = 0.0;
for (int ci = 0; ci < 3; ++ci) {
    const int pj = pcaForCur[ci];
    cv::Point3f v = pcaAxis[pj];
    const double dot = curAxis[ci].x * v.x + curAxis[ci].y * v.y + curAxis[ci].z * v.z;
    if (dot < 0.0) { v.x = -v.x; v.y = -v.y; v.z = -v.z; }
    matchedAxis[ci] = v;
    matchedVariance[ci] = pcaVariance[pj];
    const double a = std::acos(std::clamp(std::abs(dot), 0.0, 1.0));
    if (a > maxAxisAngle) maxAxisAngle = a;
}
```

**After:**
```cpp
// Strict eigenvalue-rank assignment: a = largest variance, b = middle,
// c = smallest. Greedy |dot| matching oscillated when eigenvalue
// ordering differed from current slot labeling, because matched
// variances cycled between slots each iteration. Rank assignment is
// stable — physical variance ranks are invariant to the label rotation.
cv::Point3f matchedAxis[3];
double matchedVariance[3];
double maxAxisAngle = 0.0;
for (int ci = 0; ci < 3; ++ci) {
    cv::Point3f v = pcaAxis[ci];
    // Sign-align with current slot direction for rotation continuity.
    const double dot = curAxis[ci].x * v.x + curAxis[ci].y * v.y + curAxis[ci].z * v.z;
    if (dot < 0.0) { v.x = -v.x; v.y = -v.y; v.z = -v.z; }
    matchedAxis[ci] = v;
    matchedVariance[ci] = pcaVariance[ci];
    const double ang = std::acos(std::clamp(std::abs(dot), 0.0, 1.0));
    if (ang > maxAxisAngle) maxAxisAngle = ang;
}
```

**Effect:**
- Iteration should now converge in 3-6 iters as the radii stabilize at the rank-ordered eigenvalues.
- The `axisAng` metric still measures rotation disagreement and drives the angle convergence gate.
- Slot identity (a/b/c) is determined solely by eigenvalue rank — physical tracking of a cell across frames is via rotation + radii, not slot labels.

**Open observations from the same 3-frame run** (may be downstream of oscillation; re-run after this fix):
- 1 false split: e3d03 accepted at f2 with `data_axA_primary`, costDiff=-40.28.
- 1 missed split: 12345 rejected at f3, costDiff=+4.36 (split made cost slightly worse).
- Both cells were showing chaotic eigenvalue shuffling, so stable shape may remove the spurious split geometries and give 12345 a cleaner split signal.
