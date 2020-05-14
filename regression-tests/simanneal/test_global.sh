#!/bin/bash
die() { echo "$@" >&2; exit 1;
}

echo "Testing simulated annealing global optimization"

TEST_DIR=./regression-tests/simanneal
[ -d "$TEST_DIR" ] || die "Must run from the repository's root directory!"

# delete the test output dir if exist
rm -rf $TEST_DIR/output
mkdir $TEST_DIR/output

rm -rf $TEST_DIR/bestfit
mkdir $TEST_DIR/bestfit


if python3 "./main.py" \
    --frame_first 0 \
    --frame_last 13 \
    --debug "./debug" \
    --input "./input/frame%03d.png" \
    --output "$TEST_DIR/output" \
    --bestfit "$TEST_DIR/bestfit" \
    --config "./config.json" \
    --initial "./cells.0.csv" \
    --start_temp 10 \
    --end_temp 0.1 \
    --auto_temp 0 \
    --no_parallel --global_optimization; then
    :
else
    die "Python quit unexpectedly!"
fi

python3 "$TEST_DIR/compare.py" "$TEST_DIR/expected_lineage.csv" "$TEST_DIR/output/lineage.csv" || die "compare failed"

echo "Done testing simulated annealing global optimization."
