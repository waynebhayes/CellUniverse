# OpenLab full run GT comparison, 2026-06-15 16:56:38

Run folder: `/Volumes/T9/OpenLab/Celegans_000_249_CellLumenUniverse_20260615_165638_DensityAutoBest_5f410b3_vulcan_16cpu_noTIF`
GT: `/Users/wangyiding/CellUniverse/C++/config/C.elegans developing embryo/C.elegans_initial/ground_truth/embryo_FixedGroundTruth.csv`
Match threshold: `18.0` px using `(x, y, z_interp)` against predicted `(x, y, z)`.

## Run completion

- The job did not finish frames 0-249. Slurm stderr says the process was `Killed`.
- `cells.csv` contains saved predictions for frames `0-139` only.
- Candidate graph files exist through frame `140`, but frame 140 was not saved to `cells.csv`.
- Frames `140-194` are therefore marked `NO_OUTPUT` in the comparison table.

## Summary

- Compared saved frames: `0-139`; count `140`.
- PASS frames: `4`.
- Bad saved frames: `136`.
- First bad saved frame: `f004`.
- Total missing over saved frames: `1988`.
- Total extra over saved frames: `315`.
- Error type counts: `{'pass': 4, 'both': 44, 'missing_only': 92}`.

## Bucket summary

| bucket | frames | pass | bad | no_output | total_missing | total_extra | max_missing | max_extra |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 000-020 | 21 | 4 | 17 | 0 | 41 | 1 | 5 | 1 |
| 021-040 | 20 | 0 | 20 | 0 | 172 | 2 | 17 | 2 |
| 041-085 | 45 | 0 | 45 | 0 | 859 | 3 | 26 | 1 |
| 086-120 | 35 | 0 | 35 | 0 | 735 | 47 | 41 | 5 |
| 121-139 | 19 | 0 | 19 | 0 | 181 | 262 | 14 | 18 |
| 140-194_no_output | 55 | 0 | 0 | 55 | 14887 | 0 | 362 | 0 |

## Worst missing frames

| frame | status | pred | gt | missing | extra | missing labels |
|---:|---|---:|---:|---:|---:|---|
| 192 | NO_OUTPUT | 0 | 362 | 362 | 0 | 8;9;11;12;15;16;18;19;23;24;26;27;30;31;33;34;39;40;42;43;46;47;49;50;54;55;57;58;61;62;64;65;71;72;74;75;78;79;81;82;86 |
| 193 | NO_OUTPUT | 0 | 362 | 362 | 0 | 8;9;11;12;15;16;18;19;23;24;26;27;30;31;33;34;39;40;42;43;46;47;49;50;54;55;57;58;61;62;64;65;71;72;74;75;78;79;81;82;86 |
| 194 | NO_OUTPUT | 0 | 362 | 362 | 0 | 8;9;11;12;15;16;18;19;23;24;26;27;30;31;33;34;39;40;42;43;46;47;49;50;54;55;57;58;61;62;64;65;71;72;74;75;78;79;81;82;86 |
| 190 | NO_OUTPUT | 0 | 360 | 360 | 0 | 8;9;11;12;15;16;18;19;23;24;26;27;30;31;33;34;39;40;42;43;46;47;49;50;54;55;57;58;61;62;64;65;71;72;74;75;78;79;81;82;86 |
| 191 | NO_OUTPUT | 0 | 360 | 360 | 0 | 8;9;11;12;15;16;18;19;23;24;26;27;30;31;33;34;39;40;42;43;46;47;49;50;54;55;57;58;61;62;64;65;71;72;74;75;78;79;81;82;86 |
| 186 | NO_OUTPUT | 0 | 359 | 359 | 0 | 8;9;11;12;15;16;18;19;23;24;26;27;30;31;33;34;39;40;42;43;46;47;49;50;54;55;57;58;61;62;64;65;71;72;74;75;78;79;81;82;86 |
| 187 | NO_OUTPUT | 0 | 359 | 359 | 0 | 8;9;11;12;15;16;18;19;23;24;26;27;30;31;33;34;39;40;42;43;46;47;49;50;54;55;57;58;61;62;64;65;71;72;74;75;78;79;81;82;86 |
| 188 | NO_OUTPUT | 0 | 359 | 359 | 0 | 8;9;11;12;15;16;18;19;23;24;26;27;30;31;33;34;39;40;42;43;46;47;49;50;54;55;57;58;61;62;64;65;71;72;74;75;78;79;81;82;86 |
| 189 | NO_OUTPUT | 0 | 359 | 359 | 0 | 8;9;11;12;15;16;18;19;23;24;26;27;30;31;33;34;39;40;42;43;46;47;49;50;54;55;57;58;61;62;64;65;71;72;74;75;78;79;81;82;86 |
| 185 | NO_OUTPUT | 0 | 357 | 357 | 0 | 8;9;11;12;15;16;18;19;23;24;26;27;30;31;33;34;39;40;42;43;46;47;49;50;54;55;57;58;61;62;64;65;71;72;74;75;78;79;81;82;86 |

## Worst extra frames

| frame | status | pred | gt | missing | extra | extra names |
|---:|---|---:|---:|---:|---:|---|
| 134 | FAIL_BOTH | 188 | 180 | 10 | 18 | Cell type 1_410;Cell type 1_30000;Cell type 1_2100111110;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010 |
| 135 | FAIL_BOTH | 190 | 181 | 9 | 18 | Cell type 1_410;Cell type 1_30000;Cell type 1_2100111110;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010 |
| 126 | FAIL_BOTH | 166 | 163 | 14 | 17 | Cell type 1_310111;Cell type 1_3001;Cell type 1_30000;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Ce |
| 133 | FAIL_BOTH | 186 | 180 | 11 | 17 | Cell type 1_410;Cell type 1_30000;Cell type 1_2100111110;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010 |
| 136 | FAIL_BOTH | 191 | 182 | 7 | 16 | Cell type 1_30000;Cell type 1_2100111110;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300 |
| 137 | FAIL_BOTH | 191 | 183 | 8 | 16 | Cell type 1_30000;Cell type 1_2100111110;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300 |
| 138 | FAIL_BOTH | 194 | 185 | 7 | 16 | Cell type 1_30000;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_2110101 |
| 139 | FAIL_BOTH | 195 | 186 | 7 | 16 | Cell type 1_30000;Cell type 1_21010000;Cell type 1_2101101110;Cell type 1_3100010;Cell type 1_300011;Cell type 1_2110101 |
| 128 | FAIL_BOTH | 174 | 168 | 9 | 15 | Cell type 1_310111;Cell type 1_3001;Cell type 1_30000;Cell type 1_3010;Cell type 1_21010000;Cell type 1_2101101110;Cell  |
| 127 | FAIL_BOTH | 173 | 168 | 9 | 14 | Cell type 1_310111;Cell type 1_30000;Cell type 1_3010;Cell type 1_21010001;Cell type 1_2101101110;Cell type 1_3100010;Ce |

## Files written

- `/Users/wangyiding/CellUniverse/C++/analysis/openlab_full_run_20260615_165638/frame_gt_comparison_000_194.csv`
- `/Users/wangyiding/CellUniverse/C++/analysis/openlab_full_run_20260615_165638/missing_detail_000_194.csv`
- `/Users/wangyiding/CellUniverse/C++/analysis/openlab_full_run_20260615_165638/extra_detail_000_194.csv`
- `/Users/wangyiding/CellUniverse/C++/analysis/openlab_full_run_20260615_165638/missing_label_intervals_000_194.csv`
- `/Users/wangyiding/CellUniverse/C++/analysis/openlab_full_run_20260615_165638/bucket_summary_000_194.csv`
- `/Users/wangyiding/CellUniverse/C++/analysis/openlab_full_run_20260615_165638/matched_pairs_000_194.csv`