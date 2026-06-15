# v107 density guarded CellLumen early validation

Date: 2026-06-14

Config:
`config_cell_lumen_embryo_early_v107_v105_groupGate25.yaml`

Output:
`/Users/wangyiding/CellUniverse/C++/output/CellLumen_EarlyLogicTests_20260614/v107_groupGate25_full_f000_085_noTif`

Result:

| Frames | Perfect | No miss | Missing | Extra | Max extra | Median extra | Median seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 86 | 0 | 86 | 0 | 1478 | 46 at f053 | 15 | 7.55 |

Summary:

v107 is the first full f000 to f085 standalone CellLumen sweep in this tuning
series with zero missing GT centers while substantially reducing extras versus
the older v076 high recall baseline.

Important switches:

1. Initial-prior density collapse stays enabled for sparse large-cell frames.
2. Density moment radii and frame-threshold support are used so collapsed cells
   follow the full bright body instead of only the local seed fragments.
3. `initialPriorClusterCollapseSkipAboveGroupCount: 25` exits collapse when
   linked candidate groups already look like a denser field.
4. `initialPriorClusterCollapseSkipDiameterGuardMinCells: 55` and max 63 skip
   collapse when a moderate-density frame has an oversized bridge group.
5. A narrow z-profile budget boost only fires when the post-collapse count is
   exactly 12, which recovered the f021-style upper-z evidence without globally
   raising early extras.

GT use:

Ground truth was used only for offline validation by
`audit_celegans_cell_lumen_collapse.py`. The runtime config does not read GT and
does not insert GT coordinates.

TIFF output:

No TIFF files were generated for this run.
