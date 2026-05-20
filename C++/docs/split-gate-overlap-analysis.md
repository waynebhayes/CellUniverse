# Split Gate Overlap Analysis

This note summarizes overlapping split-analysis gates in the current C++
pipeline. The goal is to make it easier to decide which gates are authoritative,
which are cheap prefilters, and which may be redundant or inconsistent.

| Overlap area | Gate / logic A | Gate / logic B | What both try to catch | Key difference | Redundancy risk |
|---|---|---|---|---|---|
| Main split vs PCA bridge split | Main split path in `trySplitCellPhased()` | PCA bridge split path | True division from elongated / two-lobed signal | Main path uses candidate burn-in, bridge gate, bio gates, overlap gate, and cost gate. PCA bridge is a separate shortcut with its own PCA/gap/cost logic. | High. PCA bridge can accept splits that main path might reject. |
| Daughter brightness | Candidate prefilter `EDGE_DIM` | Final bridge `edge_too_dim` | Daughter placed in empty or dim space | Prefilter checks local 3x3x3 brightness at candidate centers. Final bridge checks edge-zone brightness after refine/refit. | Medium. Useful two-stage check, but same biological signal. |
| Valley / dark bridge | Candidate prefilter `NO_VALLEY` | Final bridge `bridge_flat` | False split across one continuous bright cell | Prefilter uses quick projected valley estimate before ranking. Final bridge uses fuller slab-min gap/edge analysis after refine. | Medium. Intentional cheap prefilter plus final gate, but thresholds can double-reject. |
| Daughter separation | `bioCheckDaughters()` sibling buried checks | `split_daughter_overlap_gate_enabled` overlap fraction | Daughters too close, or one inside the other | Buried check tests whether a daughter center is inside the sibling ellipsoid. Overlap gate estimates body overlap fraction. | Medium-high. Overlap fraction is more general; center-inside is a cruder subset. |
| Daughter volume plausibility | `bio_combined_volume_min/max_fraction` | `bio_max_single_daughter_volume_fraction` | Fake split where daughters are too large or too small relative to parent | Combined volume checks total daughter volume. Single-daughter check rejects when one daughter inherited too much of the parent. | Medium. They constrain related volume space and need coordinated tuning. |
| Daughter size balance | `bio_daughter_size_ratio_max` | `bio_max_single_daughter_volume_fraction` | Highly asymmetric or degenerate split | Size ratio uses max-radius ratio. Single-daughter volume uses volume fraction against parent. | Low-medium. Different measurements, but both punish asymmetric candidates. |
| Cost improvement | Main split `split_cost` / `split_cost_fraction` | PCA bridge `pca_bridge_min_cost_improvement` | Split only accepted if image cost improves | Main path uses adaptive threshold with fixed plus fractional cost. PCA bridge uses a separate threshold. | High if both paths stay enabled, because acceptance economics differ. |
| Elongation trigger | Main path `P(split)` from snapshots | PCA bridge `pca_bridge_elongation_ratio` | Only elongated cells should split | Main path uses probabilistic attempt rate. PCA bridge scans after shape fitting with a hard elongation threshold. | Medium. They can disagree about when a cell is eligible. |

## Main Takeaway

The highest-risk overlap is the separate PCA bridge acceptance path. If PCA
bridge remains enabled, it can accept splits under a different set of rules than
the main split path. The cleaner architecture is to make PCA bridge produce
candidate placements for the main split path, or disable PCA bridge while tuning
the main gates.

## Keep / Remove Recommendations

| Overlap area | Recommended keep | Recommended remove / demote | Rationale | Risk if changed |
|---|---|---|---|---|
| Main split vs PCA bridge split | Keep the main split path as the authoritative acceptance path. | Demote PCA bridge from an accepting path to a candidate-proposal source, or disable it while tuning. | Main path has the richer validation stack: candidate burn-in, final bridge, daughter overlap, volume sanity, and adaptive cost. PCA bridge currently bypasses several of those checks. | Disabling PCA bridge may temporarily miss divisions that only that shortcut catches. Demoting it to proposal source is safer but requires code work. |
| Daughter brightness | Keep the final bridge `edge_too_dim` gate. | Keep candidate `EDGE_DIM` only as a ranking prefilter, not as an authoritative rejection rule. | Final bridge measures the refined daughter edge zones, which better reflects the accepted candidate. Candidate brightness is cheap and useful, but happens before refine/refit and can be stale. | Removing the prefilter may let dim-space candidates win by cost and then fail late, reducing recall or delaying correct splits by one frame. |
| Valley / dark bridge | Keep the final `bridge_flat` gate. | Keep candidate `NO_VALLEY` as soft prefilter only; consider loosening it relative to final bridge. | The final bridge has the more complete slab-min analysis after daughter refine. The prefilter is useful for candidate ranking but duplicates the same biological test. | If the prefilter is too strict, true asymmetric or close daughter splits may never reach final validation. |
| Daughter separation | Keep `split_daughter_overlap_gate_enabled` overlap fraction. | Remove or demote sibling center-inside checks in `bioCheckDaughters()`. | Fractional overlap is more expressive than checking whether one center lies inside the other ellipsoid. Center-inside is a crude subset of the overlap problem. | Removing center-inside checks is low risk if overlap fraction remains tuned; very pathological center-contained cases should still have high overlap. |
| Daughter volume plausibility | Keep both combined volume fraction and single-daughter max fraction, but tune together. | Do not remove either yet. Avoid treating either one as independently sufficient. | Combined volume catches total under/over-sized split pairs. Single-daughter max catches parent-inheritance mimics. They overlap but cover different false-split shapes. | Poor tuning can either block all real splits (`0.1` case) or allow many false splits (`0.65` case). Current value should be validated with focused resume runs. |
| Daughter size balance | Keep `bio_max_single_daughter_volume_fraction` as the primary asymmetry guard. | Consider loosening or removing `bio_daughter_size_ratio_max` only after inspecting real asymmetric divisions. | Radius ratio is a rough proxy and can punish legitimate anisotropic daughters. Volume fraction is closer to the actual biological concern. | Removing size ratio may allow odd radius-shape degeneracies unless volume and bridge gates catch them. |
| Cost improvement | Keep main split `split_cost` / `split_cost_fraction`. | Remove PCA bridge's independent cost acceptance if PCA bridge remains active; otherwise route bridge proposals through main cost gate. | One system should define acceptance economics. Main path already uses adaptive fixed-plus-fractional threshold and same bbox/cost comparison as other candidates. | If PCA bridge keeps separate cost rules, it can accept marginal or differently biased splits. |
| Elongation trigger | Keep snapshot-driven `P(split)` for the main path. | Remove PCA bridge hard elongation as an accepting-path trigger; optionally use it only to generate extra candidate proposals. | Snapshot-driven probability is integrated with the main candidate path. PCA bridge's hard threshold creates a second eligibility policy. | If hard PCA bridge trigger is removed without replacement, some very elongated late-frame cells may need higher main-path attempt probability or better candidate generation. |

## Suggested Cleanup Order

1. Make the main split path the only path that can accept a split.
2. Convert PCA bridge to emit candidate placements into the main split path.
3. Keep final bridge gates authoritative; keep candidate brightness/valley checks as cheap prefilters only.
4. Replace sibling center-inside checks with the daughter-overlap fraction gate.
5. Tune the volume gates together using resume windows, especially frames where false splits appeared after relaxing `bio_max_single_daughter_volume_fraction`.
