#!/bin/bash

die() { echo "$@" >&2; exit 1; }

echo "Testing simulated annealing with non-thresholded grayscale synthetic images"

TEST_DIR=./regression-tests/simanneal
[ -d "$TEST_DIR" ] || die "Must run from the repository's root directory!"

# create the test output dir if it doesn't exist
mkdir -p $TEST_DIR/output
rm -f $TEST_DIR/output/*

# TODO: change temp and endtemp to what works best for the new synth images
if python3 "./main.py" \
    --start 0 \
    --finish 6 \
    --graysynthetic Ture\
    --debug "./debug" \
    --input "./input_jy/original_%03d.jpg" \
    --output "./output" \
    --config "./config.json" \
    --initial "./cells.0.csv" \
    --temp 10 \
    --endtemp 0.01; then
    :
else
    die "Python quit unexpectedly!"
fi

python3 "$TEST_DIR/compare.py" "$TEST_DIR/expected_lineage.csv" "$TEST_DIR/output/lineage.csv" || die "compare failed"

echo "Done testing simulated annealing."
