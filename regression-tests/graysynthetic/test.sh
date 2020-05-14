#!/bin/bash
die() { echo "$@" >&2; exit 1;
}

echo "Testing graysynthetic"

TEST_DIR=./regression-tests/graysynthetic
[ -d "$TEST_DIR" ] || die "Must run from the repository's root directory!"

# delete the test output dir if it exist
rm -rf $TEST_DIR/output
mkdir $TEST_DIR/output

rm -rf $TEST_DIR/bestfit
mkdir $TEST_DIR/bestfit


if python3 "./main.py" \
    --frame_first 0 \
    --frame_last 13 \
    --debug "./debug" \
    --input "./input_gray/frame%03d.png" \
    --graysynthetic \
    --output "$TEST_DIR/output" \
    --bestfit "$TEST_DIR/bestfit" \
    --config "./config.json" \
    --initial "./cells.0.csv" ; then
    :
else
    die "Python quit unexpectedly!"
fi

python3 "$TEST_DIR/compare.py" "$TEST_DIR/expected_lineage.csv" "$TEST_DIR/output/lineage.csv" || die "compare failed"

echo "Done testing garysynthetic."
