#!/bin/bash
die() { echo "$@" >&2; exit 1; }

echo "Testing simulated annealing distance objective"

TEST_DIR=./regression-tests/simanneal
[ -d "$TEST_DIR" ] || die "Must run from the repository's root directory!"

# create the test output dir if it doesn't exist
mkdir -p $TEST_DIR/output
rm -f $TEST_DIR/output/*

if python "./main.py" \
    --start 0 \
    --finish 13 \
    --debug "./debug" \
    --input "./input/frame%03d.png" \
    --output "$TEST_DIR/output" \
    --config "./config.json" \
    --initial "./cells.0.csv" \
    --temp 10 \
    --endtemp 0.01 \
    --dist; then
    :
else
    die "Python quit unexpectedly!"
fi

python "$TEST_DIR/compare.py" "$TEST_DIR/expected_lineage.csv" "$TEST_DIR/output/lineage.csv"

echo "Done testing simulated annealing dist."
exit
