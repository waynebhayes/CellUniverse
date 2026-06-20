# OpenLab full run notes, 2026-06-16

This note records the problems hit while submitting the full C. elegans Cell Lumen guided Cell Universe run on UCI OpenLab, and the fixes that made the job actually start. Keep this as a local backup so the same setup issues do not waste time again.

## Target OpenLab setup

User:

```bash
yidingw6
```

Repo:

```bash
/home/yidingw6/CellUniverse_Openlab/CellUniverse_Yiding_CellLumen_SplitGuided_05272026
```

Data:

```bash
/home/yidingw6/CellUniverse_Openlab/CellUniverse_Yiding_CellLumen_SplitGuided_05272026/C++/examples/input/C.elegans_developing embryo_Fluo-N3DH-CE_Training/01/t%03d.tif
```

Main YAML:

```bash
/home/yidingw6/CellUniverse_Openlab/CellUniverse_Yiding_CellLumen_SplitGuided_05272026/C++/config/C.elegans developing embryo/Concentrated/C_elegans_DensityAuto_Best.yaml
```

Initial CSV for frame 0:

```bash
/home/yidingw6/CellUniverse_Openlab/CellUniverse_Yiding_CellLumen_SplitGuided_05272026/C++/config/C.elegans developing embryo/C.elegans_initial/initial_files/00_core_start_points/initial_embryo_0.csv
```

Slurm:

```bash
partition=openlab.p
nodelist=vulcan
cpus=16
mem=48G
```

## Problems and fixes

### 1. `sbatch` was not found

Symptom:

```text
Command 'sbatch' not found
```

Cause:

The Slurm command was not loaded in the login shell.

Fix:

```bash
module load slurm
```

### 2. The first job could not load OpenCV

Symptom:

```text
error while loading shared libraries: libopencv_stitching.so.4.5d: cannot open shared object file
```

`ldd` also showed:

```text
libopencv_imgcodecs.so.4.5d => not found
libopencv_imgproc.so.4.5d => not found
libopencv_core.so.4.5d => not found
```

Cause:

The first binary in `C++/build/celluniverse` was linked against OpenCV `4.5d`, but the compute runtime did not resolve those libraries correctly.

Important OpenCV paths found during debugging:

```bash
/lib/x86_64-linux-gnu/libopencv_stitching.so.4.5d
/lib/x86_64-linux-gnu/libopencv_core.so.4.5d
/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5d
/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5d
```

Later the safer OpenCV 4.6 path was found:

```bash
/usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.6.0
/usr/lib/x86_64-linux-gnu/libopencv_stitching.so.406
/usr/lib/x86_64-linux-gnu/libopencv_core.so.4.6.0
/usr/lib/x86_64-linux-gnu/libopencv_core.so.406
/usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.6.0
/usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.406
/usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.6.0
/usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.406
```

Best fix:

Rebuild on OpenLab/Vulcan so the binary links against the OpenLab runtime instead of relying on an older copied build.

Recommended build directory:

```bash
/home/yidingw6/CellUniverse_Openlab/CellUniverse_Yiding_CellLumen_SplitGuided_05272026/C++/build_vulcan_opencv406
```

Useful commands:

```bash
cd "/home/yidingw6/CellUniverse_Openlab/CellUniverse_Yiding_CellLumen_SplitGuided_05272026"
cmake -S "C++" -B "C++/build_vulcan_opencv406"
cmake --build "C++/build_vulcan_opencv406" -j 16
ldd "C++/build_vulcan_opencv406/celluniverse" | grep -E "opencv|not found"
```

If `ldd` prints any `not found`, do not submit the full job yet.

### 3. OpenLab `yaml-cpp` rejected duplicate YAML keys

Symptom:

```text
terminate called after throwing an instance of 'YAML::NonUniqueMapKey'
what(): yaml-cpp: error at line 398, column 9: map keys must be unique
```

Cause:

`C_elegans_DensityAuto_Best.yaml` had the same two keys twice in the same map:

```yaml
fusionSplitPriorCleanTwoRealWindowPairMinParentDistanceBalance: 0.62
fusionSplitPriorCleanTwoRealWindowPairMaxScore: 12.0
```

The duplicate values were identical. Local Mac parsing allowed this, but OpenLab `yaml-cpp` was stricter and rejected it.

Fix:

Delete the first duplicated pair and keep the later pair with the f060 comment. This is behavior equivalent because both copies had the same values.

Commit that fixed it:

```text
5f410b3
```

Verify on OpenLab:

```bash
grep -n "fusionSplitPriorCleanTwoRealWindowPairMinParentDistanceBalance\|fusionSplitPriorCleanTwoRealWindowPairMaxScore" \
"/home/yidingw6/CellUniverse_Openlab/CellUniverse_Yiding_CellLumen_SplitGuided_05272026/C++/config/C.elegans developing embryo/Concentrated/C_elegans_DensityAuto_Best.yaml"
```

Seeing the same key in different density profiles is normal. The error only happens when the same key appears twice in the same map.

### 4. Wrong or missing job script name

Symptom:

```text
sbatch: error: Unable to open file .../run_celegans_000_249_initial0_vulcan_opencv406.sh
```

Cause:

The script submitted by `sbatch` did not exist at that path. Earlier scripts had different names.

Fix:

List scripts first:

```bash
cd "/home/yidingw6/CellUniverse_Openlab/CellUniverse_Yiding_CellLumen_SplitGuided_05272026"
find openlab_jobs -maxdepth 1 -type f -name "*.sh" -print
```

If missing, recreate the intended script.

### 5. `celluniverse` was called with too few arguments

Symptom:

```text
Usage: celluniverse <firstFrame> <lastFrame> <input_pattern_or_dir_or_file> <output_dir> <config.yaml> <initial.csv>
```

Cause:

The job script passed only five runtime arguments and forgot the final `initial.csv`.

Fix:

Call:

```bash
"$BIN" 0 249 "$INPUT" "$OUT" "$YAML" "$INITIAL" 2>&1 | tee "$OUT/run_full_000_249.log"
```

### 6. Initial CSV path was wrong

Symptom:

```text
Failed to open initial CSV:
/home/yidingw6/.../C.elegans_initial/initial_embryo_0.csv
```

Cause:

The actual file is under `initial_files/00_core_start_points/`.

Correct path:

```bash
/home/yidingw6/CellUniverse_Openlab/CellUniverse_Yiding_CellLumen_SplitGuided_05272026/C++/config/C.elegans developing embryo/C.elegans_initial/initial_files/00_core_start_points/initial_embryo_0.csv
```

Fix inside job script:

```bash
INITIAL="$REPO/C++/config/C.elegans developing embryo/C.elegans_initial/initial_files/00_core_start_points/initial_embryo_0.csv"
```

### 7. Cell Lumen internal initial prior path was still the local Mac path

Symptom in run log:

```text
[CellLumen InitialPriorClusterScale] enabled=1 skipped=parse_error path=/Users/wangyiding/CellUniverse/C++/config/C.elegans developing embryo/C.elegans_initial/initial_files/00_core_start_points/initial_embryo_0.csv error="Failed to open initial CSV: ..."
```

Cause:

The main program received the correct OpenLab initial CSV, but the YAML parameter:

```yaml
initialPriorCsvPath:
```

still pointed to the local Mac path:

```bash
/Users/wangyiding/CellUniverse/...
```

Risk:

The full run can start, but early Cell Lumen initial-prior cluster collapse uses fallback behavior instead of the intended initial prior. This can affect early frame behavior. This is not a GT cheating issue; it is only a prior file path problem.

Fix options:

1. Make the code resolve `initialPriorCsvPath` relative to the repo or to the runtime initial CSV when the YAML path does not exist.
2. Or create an OpenLab-specific YAML copy where `initialPriorCsvPath` points to the OpenLab file.
3. Best long-term fix: avoid absolute local paths in shared YAML.

Current local YAML location of this setting:

```bash
C++/config/C.elegans developing embryo/Concentrated/C_elegans_DensityAuto_Best.yaml
```

Relevant lines locally:

```text
initialPriorClusterCollapseEnabled: true
initialPriorCsvPath: /Users/wangyiding/CellUniverse/C++/config/C.elegans developing embryo/C.elegans_initial/initial_files/00_core_start_points/initial_embryo_0.csv
```

### 8. Python GT comparison failed because variables were not exported

Symptom:

```text
KeyError: 'OUT'
```

Cause:

The shell variables were assigned but not exported, so `python3` could not read them from `os.environ`.

Fix:

```bash
export OUT="$(ls -td /home/yidingw6/CellUniverse_Openlab/outputs/Celegans_000_249_CellLumenUniverse_* | head -1)"
export GT="/home/yidingw6/CellUniverse_Openlab/CellUniverse_Yiding_CellLumen_SplitGuided_05272026/C++/config/C.elegans developing embryo/C.elegans_initial/ground_truth/embryo_FixedGroundTruth.csv"
```

## Known successful job start

Job:

```text
4567319
```

Status observed:

```text
4567319 openlab.p celegans yidingw6 R vulcan
```

Output folder:

```bash
/home/yidingw6/CellUniverse_Openlab/outputs/Celegans_000_249_CellLumenUniverse_20260615_165638_DensityAutoBest_5f410b3_vulcan_16cpu/
```

This job passed:

1. Slurm submission.
2. OpenCV dynamic library loading.
3. YAML duplicate key parsing.
4. Main `initial.csv` argument loading.
5. Input TIFF discovery.
6. Started running Cell Lumen on early frames.

Remaining warning:

Cell Lumen internal `initialPriorCsvPath` still pointed to the local Mac path, so the initial prior cluster collapse used fallback behavior.

## Reusable submit script skeleton

Use this only after the binary is built and `ldd` has no `not found`.

```bash
cd "/home/yidingw6/CellUniverse_Openlab/CellUniverse_Yiding_CellLumen_SplitGuided_05272026"
mkdir -p openlab_jobs

cat > openlab_jobs/run_celegans_000_249_initial0_vulcan.sh <<'EOF'
#!/bin/bash
#SBATCH --job-name=celegans_000_249_best
#SBATCH --partition=openlab.p
#SBATCH --nodelist=vulcan
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=/home/yidingw6/CellUniverse_Openlab/slurm_celegans_initial0_%j.out
#SBATCH --error=/home/yidingw6/CellUniverse_Openlab/slurm_celegans_initial0_%j.err

set -euo pipefail

REPO="/home/yidingw6/CellUniverse_Openlab/CellUniverse_Yiding_CellLumen_SplitGuided_05272026"
BIN="$REPO/C++/build_vulcan_opencv406/celluniverse"
INPUT="$REPO/C++/examples/input/C.elegans_developing embryo_Fluo-N3DH-CE_Training/01/t%03d.tif"
YAML="$REPO/C++/config/C.elegans developing embryo/Concentrated/C_elegans_DensityAuto_Best.yaml"
INITIAL="$REPO/C++/config/C.elegans developing embryo/C.elegans_initial/initial_files/00_core_start_points/initial_embryo_0.csv"

COMMIT="$(cd "$REPO" && git rev-parse --short HEAD)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTROOT="/home/yidingw6/CellUniverse_Openlab/outputs"
OUT="$OUTROOT/Celegans_000_249_CellLumenUniverse_${STAMP}_DensityAutoBest_${COMMIT}_vulcan_16cpu"

mkdir -p "$OUT"

{
  echo "Run start: $(date)"
  echo "NETID: yidingw6"
  echo "Node: $(hostname)"
  echo "Repo: $REPO"
  echo "Commit: $(cd "$REPO" && git rev-parse HEAD)"
  echo "Binary: $BIN"
  echo "Input: $INPUT"
  echo "YAML: $YAML"
  echo "Initial CSV: $INITIAL"
  echo "Output: $OUT"
  echo
  echo "ldd check:"
  ldd "$BIN" || true
  echo
} | tee "$OUT/run_metadata.txt"

if ldd "$BIN" | grep -q "not found"; then
  echo "ERROR: missing shared libraries after LD_LIBRARY_PATH setup" | tee -a "$OUT/run_metadata.txt"
  exit 127
fi

export CELLUNIVERSE_THREADS=16
export OMP_NUM_THREADS=16
export OPENCV_FOR_THREADS_NUM=16

"$BIN" 0 249 "$INPUT" "$OUT" "$YAML" "$INITIAL" 2>&1 | tee "$OUT/run_full_000_249.log"

echo "Run end: $(date)" | tee -a "$OUT/run_metadata.txt"
EOF

chmod +x openlab_jobs/run_celegans_000_249_initial0_vulcan.sh
module load slurm
sbatch openlab_jobs/run_celegans_000_249_initial0_vulcan.sh
```

## Monitoring commands

Replace `<JOB_ID>` with the submitted job id.

```bash
squeue -u yidingw6
sacct -j <JOB_ID> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList
tail -f "/home/yidingw6/CellUniverse_Openlab/slurm_celegans_initial0_<JOB_ID>.out"
tail -f "/home/yidingw6/CellUniverse_Openlab/slurm_celegans_initial0_<JOB_ID>.err"
```

Algorithm output log:

```bash
OUT="$(ls -td /home/yidingw6/CellUniverse_Openlab/outputs/Celegans_000_249_CellLumenUniverse_* | head -1)"
tail -f "$OUT/run_full_000_249.log"
```

## Quick checks before the next full run

```bash
cd "/home/yidingw6/CellUniverse_Openlab/CellUniverse_Yiding_CellLumen_SplitGuided_05272026"
git status -sb
git rev-parse --short HEAD
test -f "C++/config/C.elegans developing embryo/C.elegans_initial/initial_files/00_core_start_points/initial_embryo_0.csv" && echo "initial ok"
ldd "C++/build_vulcan_opencv406/celluniverse" | grep -E "opencv|not found"
grep -n "initialPriorCsvPath" "C++/config/C.elegans developing embryo/Concentrated/C_elegans_DensityAuto_Best.yaml"
```

If `initialPriorCsvPath` still points to `/Users/wangyiding/...`, fix that before trusting early-frame behavior on OpenLab.
