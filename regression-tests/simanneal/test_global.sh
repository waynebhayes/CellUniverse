#!/bin/bash
die() { echo "$@" >&2; exit 1;
}

echo "Testing simulated annealing global optimization"

TEST_DIR=./regression-tests/simanneal
[ -d "$TEST_DIR" ] || die "Must run from the repository's root directory!"

# create the test output dir if it doesn't exist
mkdir -p $TEST_DIR/output
rm -f $TEST_DIR/output/*

if python3 "./main.py" \
    --start 0 \
    --finish 13 \
    --debug "./debug" \
    --input "./input/frame%03d.png" \
    --output "$TEST_DIR/output" \
    --config "./config.json" \
    --initial "./cells.0.csv" \
    --temp 10 \
    --endtemp 0.01 \
    --no_parallel --global_optimization; then
    :
else
    die "Python quit unexpectedly!"
fi

python3 "$TEST_DIR/compare.py" "$TEST_DIR/expected_lineage.csv" "$TEST_DIR/output/lineage.csv" || die "compare failed"

echo "Done testing simulated annealing dist."
