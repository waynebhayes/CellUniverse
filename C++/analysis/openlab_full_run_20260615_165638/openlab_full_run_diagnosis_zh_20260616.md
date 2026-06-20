# OpenLab 0-249 Full Run Diagnosis, Focus 0-194

Run folder: `/Volumes/T9/OpenLab/Celegans_000_249_CellLumenUniverse_20260615_165638_DensityAutoBest_5f410b3_vulcan_16cpu_noTIF`

GT file: `/Users/wangyiding/CellUniverse/C++/config/C.elegans developing embryo/C.elegans_initial/ground_truth/embryo_FixedGroundTruth.csv`

Match rule: predicted `(x,y,z)` to GT `(x,y,z_interp)`, threshold `18 px`. GT is used only for audit, not for creating results.

## 1. Overall Result

Saved frames in `cells.csv`: `0-139`. Frames `140-194` have no saved output because the OpenLab process was killed around frame 140, so those rows are marked `NO_OUTPUT`, not real tracking miss.

Among saved frames, PASS frames are `4` and bad frames are `136`. The first bad saved frame is `f004` with missing `3` and extra `1`.

Total saved-frame missing count is `1988` and total saved-frame extra count is `315`. This is not a small late-frame drift. The lineage state becomes wrong immediately at the first real early split.

## 2. Bucket Summary

| bucket | frames | pass | bad | no output | missing | extra | max missing | max extra |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 000-020 | 21 | 4 | 17 | 0 | 41 | 1 | 5 | 1 |
| 021-040 | 20 | 0 | 20 | 0 | 172 | 2 | 17 | 2 |
| 041-085 | 45 | 0 | 45 | 0 | 859 | 3 | 26 | 1 |
| 086-120 | 35 | 0 | 35 | 0 | 735 | 47 | 41 | 5 |
| 121-139 | 19 | 0 | 19 | 0 | 181 | 262 | 14 | 18 |
| 140-194_no_output | 55 | 0 | 0 | 55 | 14887 | 0 | 362 | 0 |

Interpretation: `000-020` already breaks, `041-085` is severe under-splitting, `086-120` continues from a wrong lineage state, and `121-139` has both missing cells and many extra cells from accumulated false splits.

## 3. Most Important Root Cause

The first real break is f004. Cell Lumen does see the relevant bright centers, but the Universe split prior rejects them because they belong to a collapsed center cluster. This is the exact kind of hard gate problem that caused previous regressions.

For f004 in the OpenLab full run, the log shows:

```text
CellLumen Fusion CenterPrior ClusterCollapse frame=4 collapsed=4 buckets=4 radius=45
Reject CollapsedCenterClusterPair parent=Cell type 1_2 candidateIds=(5,6) sep=34.1706 rescueMinSep=35 parentShapeElong=1.41943 rescueMinParentShape=2.2 windowBoth=1
CellLumen Fusion SplitPrior GlobalSelect frame=4 ranked=2 selected=0
CellLumen SplitPrior Mode frame 4 no_cell_lumen_split_priors=1 action=skip_classic_random_split
Optimize Done frame 4 split_attempts=0 split_accepted=0 final_cells=4
```

That means the true split candidate is rejected by a very small threshold margin: separation `34.17` against required `35`. Then fallback is also blocked, so the frame stays at 4 predicted cells while GT has 6 cells.

In the previous segmented successful run, f004 accepted two Cell Lumen split priors and saved 6 cells:

```text
Split Accepted Cell type 1_1 ... candidateIds=(4,6)
Split Accepted Cell type 1_2 ... candidateIds=(5,9)
Optimize Done frame 4 split_attempts=2 split_accepted=2 final_cells=6
```

So this is not mainly a Cell Lumen detection failure. It is a split selection and gate failure after Cell Lumen already produced useful centers.

## 3.1. Missing GT To Nearest Cell Lumen Candidate

I also wrote a direct audit table for each missing GT center against the nearest `cell_lumen_high_recall` candidate:

`/Users/wangyiding/CellUniverse/C++/analysis/openlab_full_run_20260615_165638/missing_to_lumen_candidate_distance_000_139.csv`

Key examples:

| frame | missing GT label | nearest Cell Lumen candidate | distance | voxels | signal |
|---:|---:|---:|---:|---:|---:|
| 004 | 129 | 7 | 1.8992 | 4494 | 88.0451 |
| 004 | 257 | 6 | 4.9877 | 4312 | 89.009 |
| 004 | 384 | 5 | 8.2254 | 4277 | 96.1691 |
| 005 | 129 | 4 | 1.4641 | 3128 | 90.3827 |
| 018 | 609 | 4 | 1.8009 | 2847 | 93.8072 |
| 035 | 162 | 15 | 0.9270 | 2601 | 107.904 |
| 085 | 226 | 28 | 0.8482 | 6807 | 151.858 |

This supports the same conclusion: many missing cells are already visible to Cell Lumen as candidate centers. The failure is mostly in candidate commitment, split acceptance, and lineage state, not raw high recall detection.

## 4. Why Some Missing Cells Appear Back Later

Later frames can create spatially nearby centers through center repair, perturb, PCA refit, or later split candidates. This can reduce spatial missing in one frame, but it does not repair the lineage tree. Once f004 misses a real split, later recovery is only a spatial recovery, not a correct parent child history recovery.

## 5. OpenLab Portability Problem

The OpenLab run also did not exactly match local behavior because the YAML still contains a Mac absolute initial prior path. The log repeatedly says:

```text
CellLumen InitialPriorClusterScale enabled=1 skipped=parse_error path=/Users/wangyiding/.../initial_embryo_0.csv
```

The main program received the correct OpenLab initial CSV argument, but this Cell Lumen submodule still tried to read the YAML path. On OpenLab it fell back instead of reading the initial cells. This must be fixed before the next serious OpenLab proof run.

Low risk fix: if `initialPriorCsvPath` fails, use the runtime `initial.csv` argument as fallback. Also log the resolved path and whether it came from YAML or runtime initial.

## 6. What Broke Relative To Segmented Success

The earlier “success” was built by segmented runs and resume checkpoints. Some segments started from already repaired states. The one-shot run starts from frame 0 and must survive every decision in sequence. f004 shows that a later anti false split guard, especially collapsed-center pair rejection, is now too strict for the first true split. This is a clear regression pattern: a guard added to prevent internal over-splitting later is blocking true early division.

## 7. Next Code Fix Direction

1. Do not hard reject collapsed center pairs when they have future window support and are close to the separation threshold. Convert this into a soft cost or allow a rescue band around the threshold.

2. Keep hard rejection only for extreme internal texture cases: tiny separation, no dark valley, weak signal, no future support, and no parent split evidence.

3. Let fallback split logic have one safe second chance when split prior rejected all candidates only because of collapsed center status.

4. Add debug logs for every split rejection with parameter values, margins to threshold, parent age, parent shape, window evidence, signal and voxel counts. This directly addresses the “顾此失彼” problem because future failures will show which exact gate caused the error.

5. For OpenLab, fix portable initial path resolution and run a small 0-10 proof first. f004 must pass before running another 0-249 job.

## 8. Worst Saved Frames, Missing

| frame | status | pred | gt | missing | extra | missing labels short |
|---:|---|---:|---:|---:|---:|---|
| 088 | FAIL_BOTH | 46 | 83 | 41 | 4 | 13;37;44;52;59;69;76;91;99;133;140;155;164;171;179;186;196;203;...(+23) |
| 087 | FAIL_BOTH | 44 | 77 | 35 | 2 | 13;44;51;69;76;91;99;140;155;164;171;179;186;196;203;211;218;226;...(+17) |
| 086 | FAIL_BOTH | 40 | 71 | 34 | 3 | 6;51;69;76;84;91;99;155;163;178;196;203;211;218;226;292;299;307;...(+16) |
| 089 | FAIL_MISSING | 53 | 87 | 34 | 0 | 37;59;76;91;100;107;155;164;171;179;186;196;203;211;218;227;234;249;...(+16) |
| 090 | FAIL_BOTH | 55 | 88 | 34 | 1 | 37;59;76;91;100;107;155;164;171;179;186;196;203;211;218;227;249;292;...(+16) |
| 091 | FAIL_MISSING | 56 | 88 | 32 | 0 | 37;59;76;91;100;107;155;164;171;179;186;196;203;211;218;227;249;292;...(+14) |
| 092 | FAIL_MISSING | 59 | 88 | 29 | 0 | 37;59;76;100;107;155;164;171;179;186;203;211;218;227;249;292;339;388;...(+11) |
| 093 | FAIL_MISSING | 61 | 88 | 27 | 0 | 37;59;76;100;107;155;164;171;179;186;203;211;218;227;249;292;339;388;...(+9) |
| 094 | FAIL_MISSING | 61 | 88 | 27 | 0 | 37;59;76;100;107;155;164;171;179;186;203;211;218;227;249;292;339;388;...(+9) |
| 095 | FAIL_MISSING | 61 | 88 | 27 | 0 | 37;59;76;100;107;155;164;171;179;186;203;211;218;227;249;292;339;388;...(+9) |
| 085 | FAIL_BOTH | 36 | 61 | 26 | 1 | 51;68;83;99;148;155;163;178;196;203;210;226;241;291;387;403;410;496;...(+8) |
| 096 | FAIL_MISSING | 62 | 88 | 26 | 0 | 37;59;76;107;155;164;171;179;186;203;211;218;227;249;292;339;388;395;...(+8) |

## 9. Worst Saved Frames, Extra

| frame | status | pred | gt | missing | extra | extra names short |
|---:|---|---:|---:|---:|---:|---|
| 134 | FAIL_BOTH | 188 | 180 | 10 | 18 | Cell type 1_410;Cell type 1_30000;Cell type 1_2100111110;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_21101011;...(+10) |
| 135 | FAIL_BOTH | 190 | 181 | 9 | 18 | Cell type 1_410;Cell type 1_30000;Cell type 1_2100111110;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_21101011;...(+10) |
| 126 | FAIL_BOTH | 166 | 163 | 14 | 17 | Cell type 1_310111;Cell type 1_3001;Cell type 1_30000;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_211101;...(+9) |
| 133 | FAIL_BOTH | 186 | 180 | 11 | 17 | Cell type 1_410;Cell type 1_30000;Cell type 1_2100111110;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_21101011;...(+9) |
| 136 | FAIL_BOTH | 191 | 182 | 7 | 16 | Cell type 1_30000;Cell type 1_2100111110;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_21101010;Cell type 1_21011010011;...(+8) |
| 137 | FAIL_BOTH | 191 | 183 | 8 | 16 | Cell type 1_30000;Cell type 1_2100111110;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_21101011;Cell type 1_21011010011;...(+8) |
| 138 | FAIL_BOTH | 194 | 185 | 7 | 16 | Cell type 1_30000;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_21101010;Cell type 1_21011010011;Cell type 1_10011;...(+8) |
| 139 | FAIL_BOTH | 195 | 186 | 7 | 16 | Cell type 1_30000;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_21101010;Cell type 1_21011010011;Cell type 1_10011;...(+8) |
| 128 | FAIL_BOTH | 174 | 168 | 9 | 15 | Cell type 1_310111;Cell type 1_3001;Cell type 1_30000;Cell type 1_3010;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;...(+7) |
| 127 | FAIL_BOTH | 173 | 168 | 9 | 14 | Cell type 1_310111;Cell type 1_30000;Cell type 1_3010;Cell type 1_21010001;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_21101011;...(+6) |
| 129 | FAIL_BOTH | 178 | 174 | 10 | 14 | Cell type 1_30000;Cell type 1_3010;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_21101011;Cell type 1_21011010011;...(+6) |
| 131 | FAIL_BOTH | 182 | 178 | 9 | 13 | Cell type 1_30000;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_21101011;Cell type 1_21011010011;Cell type 1_10011;...(+5) |

## 10. Full Per Frame Count Table

| frame | status | pred | gt | matched | missing | extra | split accepted | Cell Lumen candidates | missing labels short | extra names short |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 000 | PASS | 4 | 4 | 4 | 0 | 0 | 0 | 9 |  |  |
| 001 | PASS | 4 | 4 | 4 | 0 | 0 | 0 | 9 |  |  |
| 002 | PASS | 4 | 4 | 4 | 0 | 0 | 0 | 9 |  |  |
| 003 | PASS | 4 | 4 | 4 | 0 | 0 | 0 | 8 |  |  |
| 004 | FAIL_BOTH | 4 | 6 | 3 | 3 | 1 | 0 | 12 | 129;257;384 | Cell type 1_2 |
| 005 | FAIL_MISSING | 5 | 6 | 5 | 1 | 0 | 1 | 11 | 129 |  |
| 006 | FAIL_MISSING | 5 | 6 | 5 | 1 | 0 | 0 | 11 | 129 |  |
| 007 | FAIL_MISSING | 5 | 7 | 5 | 2 | 0 | 0 | 13 | 129;512 |  |
| 008 | FAIL_MISSING | 5 | 7 | 5 | 2 | 0 | 0 | 12 | 129;609 |  |
| 009 | FAIL_MISSING | 5 | 7 | 5 | 2 | 0 | 0 | 12 | 129;609 |  |
| 010 | FAIL_MISSING | 5 | 7 | 5 | 2 | 0 | 0 | 12 | 129;609 |  |
| 011 | FAIL_MISSING | 6 | 8 | 6 | 2 | 0 | 1 | 13 | 129;609 |  |
| 012 | FAIL_MISSING | 6 | 8 | 6 | 2 | 0 | 0 | 12 | 129;609 |  |
| 013 | FAIL_MISSING | 6 | 8 | 6 | 2 | 0 | 0 | 17 | 129;609 |  |
| 014 | FAIL_MISSING | 6 | 8 | 6 | 2 | 0 | 0 | 19 | 129;609 |  |
| 015 | FAIL_MISSING | 6 | 8 | 6 | 2 | 0 | 0 | 20 | 129;609 |  |
| 016 | FAIL_MISSING | 6 | 8 | 6 | 2 | 0 | 0 | 12 | 129;609 |  |
| 017 | FAIL_MISSING | 6 | 8 | 6 | 2 | 0 | 0 | 14 | 129;609 |  |
| 018 | FAIL_MISSING | 6 | 10 | 6 | 4 | 0 | 0 | 15 | 66;129;448;609 |  |
| 019 | FAIL_MISSING | 7 | 12 | 7 | 5 | 0 | 1 | 17 | 66;130;193;448;609 |  |
| 020 | FAIL_MISSING | 7 | 12 | 7 | 5 | 0 | 0 | 17 | 66;130;193;448;609 |  |
| 021 | FAIL_MISSING | 7 | 12 | 7 | 5 | 0 | 0 | 24 | 66;130;193;448;609 |  |
| 022 | FAIL_MISSING | 7 | 12 | 7 | 5 | 0 | 0 | 24 | 66;130;193;448;609 |  |
| 023 | FAIL_MISSING | 7 | 12 | 7 | 5 | 0 | 0 | 24 | 66;130;193;448;609 |  |
| 024 | FAIL_MISSING | 7 | 13 | 7 | 6 | 0 | 0 | 18 | 66;130;193;448;564;609 |  |
| 025 | FAIL_MISSING | 7 | 14 | 7 | 7 | 0 | 0 | 18 | 66;130;193;448;564;610;625 |  |
| 026 | FAIL_MISSING | 7 | 14 | 7 | 7 | 0 | 0 | 22 | 66;130;193;448;564;610;625 |  |
| 027 | FAIL_MISSING | 7 | 14 | 7 | 7 | 0 | 0 | 21 | 66;130;193;448;564;610;625 |  |
| 028 | FAIL_MISSING | 7 | 14 | 7 | 7 | 0 | 0 | 32 | 66;130;193;448;564;610;625 |  |
| 029 | FAIL_MISSING | 8 | 15 | 8 | 7 | 0 | 1 | 32 | 66;130;193;448;564;610;625 |  |
| 030 | FAIL_MISSING | 8 | 15 | 8 | 7 | 0 | 0 | 35 | 66;130;193;448;564;610;625 |  |
| 031 | FAIL_MISSING | 8 | 15 | 8 | 7 | 0 | 0 | 34 | 66;130;193;448;564;610;625 |  |
| 032 | FAIL_MISSING | 8 | 15 | 8 | 7 | 0 | 0 | 35 | 66;130;193;448;564;610;625 |  |
| 033 | FAIL_MISSING | 8 | 15 | 8 | 7 | 0 | 0 | 20 | 66;130;193;448;564;610;625 |  |
| 034 | FAIL_MISSING | 8 | 15 | 8 | 7 | 0 | 0 | 20 | 66;130;193;448;564;610;625 |  |
| 035 | FAIL_BOTH | 9 | 24 | 7 | 17 | 2 | 1 | 37 | 4;35;67;98;131;162;194;225;290;386;449;480;...(+5) | Cell type 1_1;Cell type 1_30 |
| 036 | FAIL_MISSING | 11 | 24 | 11 | 13 | 0 | 2 | 36 | 67;98;131;162;194;225;290;386;449;480;564;610;...(+1) |  |
| 037 | FAIL_MISSING | 11 | 24 | 11 | 13 | 0 | 0 | 36 | 67;98;131;162;194;225;290;386;449;480;564;610;...(+1) |  |
| 038 | FAIL_MISSING | 11 | 24 | 11 | 13 | 0 | 0 | 40 | 67;98;131;162;194;225;290;386;449;480;564;610;...(+1) |  |
| 039 | FAIL_MISSING | 11 | 24 | 11 | 13 | 0 | 0 | 40 | 67;98;131;162;194;225;290;386;449;480;564;610;...(+1) |  |
| 040 | FAIL_MISSING | 12 | 24 | 12 | 12 | 0 | 1 | 39 | 67;98;131;162;194;225;290;386;449;480;564;625 |  |
| 041 | FAIL_MISSING | 12 | 24 | 12 | 12 | 0 | 0 | 41 | 67;98;131;162;194;225;290;386;449;480;564;625 |  |
| 042 | FAIL_MISSING | 12 | 24 | 12 | 12 | 0 | 0 | 38 | 67;98;131;162;194;225;290;386;449;480;564;625 |  |
| 043 | FAIL_MISSING | 12 | 24 | 12 | 12 | 0 | 0 | 39 | 67;98;131;162;194;225;290;386;449;480;564;625 |  |
| 044 | FAIL_MISSING | 12 | 24 | 12 | 12 | 0 | 0 | 36 | 67;98;131;162;194;225;290;386;449;480;564;625 |  |
| 045 | FAIL_MISSING | 12 | 26 | 12 | 14 | 0 | 0 | 46 | 67;98;131;162;194;225;290;386;449;480;541;565;...(+2) |  |
| 046 | FAIL_MISSING | 12 | 26 | 12 | 14 | 0 | 0 | 62 | 67;98;131;162;194;225;290;386;449;480;541;565;...(+2) |  |
| 047 | FAIL_MISSING | 12 | 26 | 12 | 14 | 0 | 0 | 57 | 67;98;131;162;194;225;290;386;449;480;541;565;...(+2) |  |
| 048 | FAIL_MISSING | 12 | 26 | 12 | 14 | 0 | 0 | 64 | 67;98;131;162;194;225;290;386;449;480;541;565;...(+2) |  |
| 049 | FAIL_MISSING | 12 | 26 | 12 | 14 | 0 | 0 | 61 | 67;98;131;162;194;225;290;386;449;480;541;565;...(+2) |  |
| 050 | FAIL_MISSING | 12 | 26 | 12 | 14 | 0 | 0 | 62 | 67;98;131;162;194;225;290;386;449;480;541;565;...(+2) |  |
| 051 | FAIL_MISSING | 12 | 26 | 12 | 14 | 0 | 0 | 69 | 67;98;131;162;194;225;290;386;449;480;541;565;...(+2) |  |
| 052 | FAIL_MISSING | 14 | 28 | 14 | 14 | 0 | 2 | 61 | 67;98;131;162;194;225;290;386;449;480;541;565;...(+2) |  |
| 053 | FAIL_MISSING | 14 | 28 | 14 | 14 | 0 | 0 | 68 | 67;98;131;162;194;225;290;386;449;480;541;565;...(+2) |  |
| 054 | FAIL_MISSING | 14 | 28 | 14 | 14 | 0 | 0 | 69 | 67;98;131;162;194;225;290;386;449;480;541;565;...(+2) |  |
| 055 | FAIL_MISSING | 15 | 28 | 15 | 13 | 0 | 1 | 70 | 67;98;131;162;194;225;290;386;480;541;565;590;...(+1) |  |
| 056 | FAIL_MISSING | 15 | 28 | 15 | 13 | 0 | 0 | 74 | 67;98;131;162;194;225;290;386;480;541;565;590;...(+1) |  |
| 057 | FAIL_MISSING | 16 | 28 | 16 | 12 | 0 | 1 | 66 | 67;98;162;194;225;290;386;480;541;565;590;625 |  |
| 058 | FAIL_MISSING | 17 | 30 | 17 | 13 | 0 | 1 | 63 | 67;98;147;162;194;225;290;386;480;541;565;590;...(+1) |  |
| 059 | FAIL_BOTH | 18 | 37 | 17 | 20 | 1 | 1 | 58 | 68;83;98;147;163;178;194;225;291;306;387;402;...(+8) | Cell type 1_201 |
| 060 | FAIL_BOTH | 20 | 43 | 19 | 24 | 1 | 2 | 57 | 51;68;83;99;114;147;163;178;195;210;225;291;...(+12) | Cell type 1_201 |
| 061 | FAIL_MISSING | 21 | 45 | 21 | 24 | 0 | 1 | 65 | 51;68;83;99;114;147;163;178;195;210;226;241;...(+12) |  |
| 062 | FAIL_MISSING | 22 | 45 | 22 | 23 | 0 | 1 | 64 | 51;68;83;99;114;147;163;178;195;210;226;241;...(+11) |  |
| 063 | FAIL_MISSING | 22 | 45 | 22 | 23 | 0 | 0 | 64 | 51;68;83;99;114;147;163;178;195;210;226;241;...(+11) |  |
| 064 | FAIL_MISSING | 23 | 46 | 23 | 23 | 0 | 1 | 66 | 51;68;83;99;114;147;163;178;195;210;226;241;...(+11) |  |
| 065 | FAIL_MISSING | 23 | 46 | 23 | 23 | 0 | 0 | 69 | 51;68;83;99;114;147;163;178;195;210;226;241;...(+11) |  |
| 066 | FAIL_MISSING | 23 | 47 | 23 | 24 | 0 | 0 | 75 | 51;68;83;99;114;147;163;178;195;210;226;241;...(+12) |  |
| 067 | FAIL_MISSING | 23 | 47 | 23 | 24 | 0 | 0 | 70 | 51;68;83;99;114;147;163;178;195;210;226;241;...(+12) |  |
| 068 | FAIL_MISSING | 24 | 48 | 24 | 24 | 0 | 1 | 72 | 51;68;83;99;114;147;163;178;195;210;226;241;...(+12) |  |
| 069 | FAIL_MISSING | 25 | 49 | 25 | 24 | 0 | 1 | 69 | 51;68;83;99;114;147;163;178;195;210;226;241;...(+12) |  |
| 070 | FAIL_MISSING | 26 | 50 | 26 | 24 | 0 | 1 | 68 | 51;68;83;99;114;147;163;178;195;210;226;241;...(+12) |  |
| 071 | FAIL_MISSING | 26 | 51 | 26 | 25 | 0 | 0 | 70 | 51;68;83;99;114;147;163;178;195;210;226;241;...(+13) |  |
| 072 | FAIL_MISSING | 27 | 51 | 27 | 24 | 0 | 1 | 82 | 51;68;83;99;147;163;178;195;210;226;241;291;...(+12) |  |
| 073 | FAIL_MISSING | 27 | 51 | 27 | 24 | 0 | 0 | 72 | 51;68;83;99;147;163;178;195;210;226;241;291;...(+12) |  |
| 074 | FAIL_MISSING | 28 | 51 | 28 | 23 | 0 | 1 | 73 | 51;68;83;99;147;163;178;195;210;226;241;291;...(+11) |  |
| 075 | FAIL_MISSING | 28 | 51 | 28 | 23 | 0 | 0 | 75 | 51;68;83;99;147;163;178;195;210;226;241;291;...(+11) |  |
| 076 | FAIL_MISSING | 28 | 51 | 28 | 23 | 0 | 0 | 73 | 51;68;83;99;147;163;178;195;210;226;241;291;...(+11) |  |
| 077 | FAIL_MISSING | 28 | 51 | 28 | 23 | 0 | 0 | 82 | 51;68;83;99;147;163;178;195;210;226;241;291;...(+11) |  |
| 078 | FAIL_MISSING | 29 | 51 | 29 | 22 | 0 | 1 | 73 | 51;68;83;99;147;163;178;195;210;226;241;291;...(+10) |  |
| 079 | FAIL_MISSING | 29 | 51 | 29 | 22 | 0 | 0 | 65 | 51;68;83;99;147;163;178;195;210;226;241;291;...(+10) |  |
| 080 | FAIL_MISSING | 32 | 53 | 32 | 21 | 0 | 3 | 79 | 51;68;83;99;147;163;178;195;210;226;241;291;...(+9) |  |
| 081 | FAIL_MISSING | 32 | 53 | 32 | 21 | 0 | 0 | 77 | 51;68;83;99;147;163;178;195;210;226;241;291;...(+9) |  |
| 082 | FAIL_MISSING | 33 | 54 | 33 | 21 | 0 | 1 | 82 | 51;68;83;99;147;163;178;195;210;226;241;291;...(+9) |  |
| 083 | FAIL_MISSING | 33 | 54 | 33 | 21 | 0 | 0 | 79 | 51;68;83;99;147;163;178;195;210;226;241;291;...(+9) |  |
| 084 | FAIL_MISSING | 34 | 55 | 34 | 21 | 0 | 1 | 82 | 51;68;83;99;147;163;178;195;210;226;241;291;...(+9) |  |
| 085 | FAIL_BOTH | 36 | 61 | 35 | 26 | 1 | 2 | 78 | 51;68;83;99;148;155;163;178;196;203;210;226;...(+14) | Cell type 1_31111 |
| 086 | FAIL_BOTH | 40 | 71 | 37 | 34 | 3 | 4 | 88 | 6;51;69;76;84;91;99;155;163;178;196;203;...(+22) | Cell type 1_210010;Cell type 1_210100;Cell type 1_2101111 |
| 087 | FAIL_BOTH | 44 | 77 | 42 | 35 | 2 | 4 | 94 | 13;44;51;69;76;91;99;140;155;164;171;179;...(+23) | Cell type 1_21100;Cell type 1_2101111 |
| 088 | FAIL_BOTH | 46 | 83 | 42 | 41 | 4 | 2 | 97 | 13;37;44;52;59;69;76;91;99;133;140;155;...(+29) | Cell type 1_101;Cell type 1_21100;Cell type 1_1001;Cell type 1_201001 |
| 089 | FAIL_MISSING | 53 | 87 | 53 | 34 | 0 | 7 | 98 | 37;59;76;91;100;107;155;164;171;179;186;196;...(+22) |  |
| 090 | FAIL_BOTH | 55 | 88 | 54 | 34 | 1 | 2 | 116 | 37;59;76;91;100;107;155;164;171;179;186;196;...(+22) | Cell type 1_20111 |
| 091 | FAIL_MISSING | 56 | 88 | 56 | 32 | 0 | 1 | 121 | 37;59;76;91;100;107;155;164;171;179;186;196;...(+20) |  |
| 092 | FAIL_MISSING | 59 | 88 | 59 | 29 | 0 | 3 |  | 37;59;76;100;107;155;164;171;179;186;203;211;...(+17) |  |
| 093 | FAIL_MISSING | 61 | 88 | 61 | 27 | 0 | 2 |  | 37;59;76;100;107;155;164;171;179;186;203;211;...(+15) |  |
| 094 | FAIL_MISSING | 61 | 88 | 61 | 27 | 0 | 0 |  | 37;59;76;100;107;155;164;171;179;186;203;211;...(+15) |  |
| 095 | FAIL_MISSING | 61 | 88 | 61 | 27 | 0 | 0 |  | 37;59;76;100;107;155;164;171;179;186;203;211;...(+15) |  |
| 096 | FAIL_MISSING | 62 | 88 | 62 | 26 | 0 | 1 |  | 37;59;76;107;155;164;171;179;186;203;211;218;...(+14) |  |
| 097 | FAIL_MISSING | 66 | 90 | 66 | 24 | 0 | 4 |  | 37;59;76;107;155;171;179;186;203;211;218;227;...(+12) |  |
| 098 | FAIL_BOTH | 70 | 92 | 68 | 24 | 2 | 4 |  | 37;76;107;155;171;179;186;203;211;218;227;249;...(+12) | Cell type 1_30000;Cell type 1_3110110 |
| 099 | FAIL_BOTH | 72 | 93 | 70 | 23 | 2 | 2 |  | 37;59;76;155;171;179;186;203;211;218;227;249;...(+11) | Cell type 1_30000;Cell type 1_3110110 |
| 100 | FAIL_BOTH | 72 | 93 | 69 | 24 | 3 | 0 |  | 37;76;155;171;179;186;203;211;218;227;249;292;...(+12) | Cell type 1_3001;Cell type 1_30000;Cell type 1_3110110 |
| 101 | FAIL_BOTH | 74 | 95 | 69 | 26 | 5 | 2 |  | 37;76;155;171;179;186;203;211;218;227;249;388;...(+14) | Cell type 1_3001;Cell type 1_410;Cell type 1_30000;Cell type 1_3110110;Cell type 1_21011001 |
| 102 | FAIL_BOTH | 75 | 97 | 72 | 25 | 3 | 1 |  | 37;76;155;171;179;186;203;211;218;227;249;388;...(+13) | Cell type 1_410;Cell type 1_30001;Cell type 1_3110110 |
| 103 | FAIL_BOTH | 76 | 97 | 74 | 23 | 2 | 1 |  | 37;155;171;179;186;203;211;218;227;249;388;395;...(+11) | Cell type 1_410;Cell type 1_3110110 |
| 104 | FAIL_BOTH | 76 | 97 | 74 | 23 | 2 | 0 |  | 37;155;171;179;186;203;211;218;227;249;388;395;...(+11) | Cell type 1_410;Cell type 1_3110110 |
| 105 | FAIL_BOTH | 79 | 98 | 77 | 21 | 2 | 3 |  | 37;155;171;179;186;203;211;218;249;388;395;403;...(+9) | Cell type 1_410;Cell type 1_3110110 |
| 106 | FAIL_BOTH | 81 | 100 | 80 | 20 | 1 | 2 |  | 37;155;171;179;186;203;211;218;249;388;395;451;...(+8) | Cell type 1_3110110 |
| 107 | FAIL_BOTH | 83 | 100 | 82 | 18 | 1 | 2 |  | 171;179;186;203;211;218;249;395;451;567;574;580;...(+6) | Cell type 1_3110110 |
| 108 | FAIL_MISSING | 84 | 101 | 84 | 17 | 0 | 2 |  | 171;179;186;203;211;218;249;395;451;574;580;587;...(+5) |  |
| 109 | FAIL_MISSING | 86 | 102 | 86 | 16 | 0 | 2 |  | 171;179;186;203;211;218;249;395;451;574;580;587;...(+4) |  |
| 110 | FAIL_MISSING | 86 | 102 | 86 | 16 | 0 | 0 |  | 171;179;186;203;211;218;249;395;451;574;580;587;...(+4) |  |
| 111 | FAIL_MISSING | 91 | 103 | 91 | 12 | 0 | 5 |  | 171;179;203;218;451;574;580;587;597;603;712;715 |  |
| 112 | FAIL_MISSING | 93 | 105 | 93 | 12 | 0 | 2 |  | 171;179;218;451;574;580;587;597;603;684;712;715 |  |
| 113 | FAIL_BOTH | 95 | 105 | 94 | 11 | 1 | 2 |  | 179;218;451;574;580;587;597;603;651;712;715 | Cell type 1_31100 |
| 114 | FAIL_BOTH | 97 | 105 | 96 | 9 | 1 | 2 |  | 218;574;580;587;597;603;705;712;715 | Cell type 1_3011 |
| 115 | FAIL_MISSING | 99 | 106 | 99 | 7 | 0 | 2 |  | 574;580;587;597;603;712;715 |  |
| 116 | FAIL_MISSING | 99 | 106 | 99 | 7 | 0 | 0 |  | 574;580;587;597;603;712;715 |  |
| 117 | FAIL_BOTH | 101 | 108 | 100 | 8 | 1 | 2 |  | 574;580;587;597;603;705;712;715 | Cell type 1_3011 |
| 118 | FAIL_BOTH | 106 | 112 | 103 | 9 | 3 | 5 |  | 574;580;587;597;603;652;677;712;715 | Cell type 1_310001;Cell type 1_4110;Cell type 1_21010001 |
| 119 | FAIL_BOTH | 114 | 118 | 110 | 8 | 4 | 8 |  | 350;430;574;580;597;603;712;715 | Cell type 1_4111;Cell type 1_2010100;Cell type 1_21010001;Cell type 1_3100011 |
| 120 | FAIL_BOTH | 127 | 129 | 123 | 6 | 4 | 13 |  | 350;430;580;603;712;715 | Cell type 1_4110;Cell type 1_2010100;Cell type 1_21010001;Cell type 1_3100010 |
| 121 | FAIL_BOTH | 135 | 139 | 129 | 10 | 6 | 9 |  | 14;48;126;350;388;430;580;603;712;715 | Cell type 1_2010100;Cell type 1_21010001;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_21101011;Cell type 1_21011010011 |
| 122 | FAIL_BOTH | 143 | 145 | 133 | 12 | 10 | 8 |  | 48;88;108;111;123;350;430;580;603;648;712;715 | Cell type 1_20010;Cell type 1_210110101;Cell type 1_21011100;Cell type 1_31101111;Cell type 1_21010001;Cell type 1_2101101110;...(+4) |
| 123 | FAIL_BOTH | 148 | 151 | 138 | 13 | 10 | 5 |  | 48;88;92;108;111;168;350;430;580;603;606;712;...(+1) | Cell type 1_21011100;Cell type 1_21010001;Cell type 1_2101101110;Cell type 1_2101101111;Cell type 1_3100010;Cell type 1_300011;...(+4) |
| 124 | FAIL_BOTH | 158 | 157 | 148 | 9 | 10 | 10 |  | 196;350;580;603;606;648;712;715;720 | Cell type 1_3001;Cell type 1_10110;Cell type 1_31101111;Cell type 1_21010001;Cell type 1_2101101110;Cell type 1_3100010;...(+4) |
| 125 | FAIL_BOTH | 162 | 161 | 151 | 10 | 11 | 4 |  | 196;350;580;603;606;708;712;715;719;720 | Cell type 1_3001;Cell type 1_10110;Cell type 1_30000;Cell type 1_3010;Cell type 1_21010001;Cell type 1_2101101110;...(+5) |
| 126 | FAIL_BOTH | 166 | 163 | 149 | 14 | 17 | 4 |  | 88;280;308;315;350;470;580;603;606;696;712;715;...(+2) | Cell type 1_310111;Cell type 1_3001;Cell type 1_30000;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;...(+11) |
| 127 | FAIL_BOTH | 173 | 168 | 159 | 9 | 14 | 7 |  | 108;470;580;603;696;708;712;715;719 | Cell type 1_310111;Cell type 1_30000;Cell type 1_3010;Cell type 1_21010001;Cell type 1_2101101110;Cell type 1_3100010;...(+8) |
| 128 | FAIL_BOTH | 174 | 168 | 159 | 9 | 15 | 1 |  | 108;311;580;603;708;712;715;719;720 | Cell type 1_310111;Cell type 1_3001;Cell type 1_30000;Cell type 1_3010;Cell type 1_21010000;Cell type 1_2101101110;...(+9) |
| 129 | FAIL_BOTH | 178 | 174 | 164 | 10 | 14 | 4 |  | 38;108;311;581;584;603;708;712;715;719 | Cell type 1_30000;Cell type 1_3010;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;...(+8) |
| 130 | FAIL_BOTH | 180 | 176 | 168 | 8 | 12 | 2 |  | 108;571;581;603;692;712;715;719 | Cell type 1_30000;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_21101011;...(+6) |
| 131 | FAIL_BOTH | 182 | 178 | 169 | 9 | 13 | 3 |  | 108;568;571;581;603;692;712;715;719 | Cell type 1_30000;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_21101011;...(+7) |
| 132 | FAIL_BOTH | 183 | 179 | 170 | 9 | 13 | 1 |  | 48;108;571;581;603;692;712;715;719 | Cell type 1_30000;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_21101011;...(+7) |
| 133 | FAIL_BOTH | 186 | 180 | 169 | 11 | 17 | 4 |  | 48;108;535;538;571;581;592;692;712;715;719 | Cell type 1_410;Cell type 1_30000;Cell type 1_2100111110;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;...(+11) |
| 134 | FAIL_BOTH | 188 | 180 | 170 | 10 | 18 | 2 |  | 48;101;108;581;592;661;692;712;715;719 | Cell type 1_410;Cell type 1_30000;Cell type 1_2100111110;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;...(+12) |
| 135 | FAIL_BOTH | 190 | 181 | 172 | 9 | 18 | 2 |  | 101;539;540;581;592;692;712;715;719 | Cell type 1_410;Cell type 1_30000;Cell type 1_2100111110;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;...(+12) |
| 136 | FAIL_BOTH | 191 | 182 | 175 | 7 | 16 | 1 |  | 293;581;592;692;712;715;719 | Cell type 1_30000;Cell type 1_2100111110;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;...(+10) |
| 137 | FAIL_BOTH | 191 | 183 | 175 | 8 | 16 | 0 |  | 293;554;581;592;692;712;715;719 | Cell type 1_30000;Cell type 1_2100111110;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;...(+10) |
| 138 | FAIL_BOTH | 194 | 185 | 178 | 7 | 16 | 3 |  | 293;581;593;692;712;715;719 | Cell type 1_30000;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_21101010;...(+10) |
| 139 | FAIL_BOTH | 195 | 186 | 179 | 7 | 16 | 1 |  | 48;581;593;692;712;715;719 | Cell type 1_30000;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_21101010;...(+10) |
| 140 | NO_OUTPUT | 0 | 186 | 0 | 186 | 0 | 0 | 198 | 7;10;14;17;22;25;29;32;38;41;45;48;...(+174) |  |
| 141 | NO_OUTPUT | 0 | 186 | 0 | 186 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+174) |  |
| 142 | NO_OUTPUT | 0 | 186 | 0 | 186 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+174) |  |
| 143 | NO_OUTPUT | 0 | 187 | 0 | 187 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+175) |  |
| 144 | NO_OUTPUT | 0 | 187 | 0 | 187 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+175) |  |
| 145 | NO_OUTPUT | 0 | 188 | 0 | 188 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+176) |  |
| 146 | NO_OUTPUT | 0 | 190 | 0 | 190 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+178) |  |
| 147 | NO_OUTPUT | 0 | 191 | 0 | 191 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+179) |  |
| 148 | NO_OUTPUT | 0 | 192 | 0 | 192 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+180) |  |
| 149 | NO_OUTPUT | 0 | 192 | 0 | 192 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+180) |  |
| 150 | NO_OUTPUT | 0 | 192 | 0 | 192 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+180) |  |
| 151 | NO_OUTPUT | 0 | 192 | 0 | 192 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+180) |  |
| 152 | NO_OUTPUT | 0 | 193 | 0 | 193 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+181) |  |
| 153 | NO_OUTPUT | 0 | 193 | 0 | 193 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+181) |  |
| 154 | NO_OUTPUT | 0 | 195 | 0 | 195 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+183) |  |
| 155 | NO_OUTPUT | 0 | 197 | 0 | 197 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+185) |  |
| 156 | NO_OUTPUT | 0 | 200 | 0 | 200 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+188) |  |
| 157 | NO_OUTPUT | 0 | 200 | 0 | 200 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+188) |  |
| 158 | NO_OUTPUT | 0 | 201 | 0 | 201 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+189) |  |
| 159 | NO_OUTPUT | 0 | 205 | 0 | 205 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+193) |  |
| 160 | NO_OUTPUT | 0 | 206 | 0 | 206 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+194) |  |
| 161 | NO_OUTPUT | 0 | 211 | 0 | 211 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+199) |  |
| 162 | NO_OUTPUT | 0 | 215 | 0 | 215 | 0 |  |  | 7;10;14;17;22;25;29;32;38;41;45;48;...(+203) |  |
| 163 | NO_OUTPUT | 0 | 227 | 0 | 227 | 0 |  |  | 7;10;14;18;19;22;25;29;32;38;41;45;...(+215) |  |
| 164 | NO_OUTPUT | 0 | 232 | 0 | 232 | 0 |  |  | 7;10;14;18;19;22;25;29;33;34;38;41;...(+220) |  |
| 165 | NO_OUTPUT | 0 | 240 | 0 | 240 | 0 |  |  | 7;10;14;18;19;22;25;29;33;34;38;41;...(+228) |  |
| 166 | NO_OUTPUT | 0 | 256 | 0 | 256 | 0 |  |  | 7;10;15;16;18;19;22;25;29;33;34;38;...(+244) |  |
| 167 | NO_OUTPUT | 0 | 268 | 0 | 268 | 0 |  |  | 8;9;10;15;16;18;19;22;25;30;31;33;...(+256) |  |
| 168 | NO_OUTPUT | 0 | 278 | 0 | 278 | 0 |  |  | 8;9;10;15;16;18;19;22;25;30;31;33;...(+266) |  |
| 169 | NO_OUTPUT | 0 | 287 | 0 | 287 | 0 |  |  | 8;9;11;12;15;16;18;19;22;25;30;31;...(+275) |  |
| 170 | NO_OUTPUT | 0 | 299 | 0 | 299 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+287) |  |
| 171 | NO_OUTPUT | 0 | 306 | 0 | 306 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+294) |  |
| 172 | NO_OUTPUT | 0 | 316 | 0 | 316 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+304) |  |
| 173 | NO_OUTPUT | 0 | 317 | 0 | 317 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+305) |  |
| 174 | NO_OUTPUT | 0 | 324 | 0 | 324 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+312) |  |
| 175 | NO_OUTPUT | 0 | 329 | 0 | 329 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+317) |  |
| 176 | NO_OUTPUT | 0 | 334 | 0 | 334 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+322) |  |
| 177 | NO_OUTPUT | 0 | 340 | 0 | 340 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+328) |  |
| 178 | NO_OUTPUT | 0 | 342 | 0 | 342 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+330) |  |
| 179 | NO_OUTPUT | 0 | 346 | 0 | 346 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+334) |  |
| 180 | NO_OUTPUT | 0 | 348 | 0 | 348 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+336) |  |
| 181 | NO_OUTPUT | 0 | 350 | 0 | 350 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+338) |  |
| 182 | NO_OUTPUT | 0 | 352 | 0 | 352 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+340) |  |
| 183 | NO_OUTPUT | 0 | 356 | 0 | 356 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+344) |  |
| 184 | NO_OUTPUT | 0 | 356 | 0 | 356 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+344) |  |
| 185 | NO_OUTPUT | 0 | 357 | 0 | 357 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+345) |  |
| 186 | NO_OUTPUT | 0 | 359 | 0 | 359 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+347) |  |
| 187 | NO_OUTPUT | 0 | 359 | 0 | 359 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+347) |  |
| 188 | NO_OUTPUT | 0 | 359 | 0 | 359 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+347) |  |
| 189 | NO_OUTPUT | 0 | 359 | 0 | 359 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+347) |  |
| 190 | NO_OUTPUT | 0 | 360 | 0 | 360 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+348) |  |
| 191 | NO_OUTPUT | 0 | 360 | 0 | 360 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+348) |  |
| 192 | NO_OUTPUT | 0 | 362 | 0 | 362 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+350) |  |
| 193 | NO_OUTPUT | 0 | 362 | 0 | 362 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+350) |  |
| 194 | NO_OUTPUT | 0 | 362 | 0 | 362 | 0 |  |  | 8;9;11;12;15;16;18;19;23;24;26;27;...(+350) |  |

Full exact CSV files are in the same folder. Use the CSV for machine sorting and this markdown for review.
