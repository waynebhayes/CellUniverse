#!/bin/bash
die() { echo "$@" >&2; exit 1;
}

echo "Testing binary local simulated annealing"

[ -d "$REG_DIR" ] || die "Must run from the repository's root directory!"

# create the test output dir if it doesn't exist
rm -rf $REG_DIR/output
mkdir  $REG_DIR/output

rm -rf $REG_DIR/bestfit
mkdir $REG_DIR/bestfit

if python3 "./src/main.py" \
    --frame_first 0 \
    --frame_last 13 \
    --debug "./debug" \
    --input "./examples/canonical/input/gray/frame%03d.png" \
    --output "$REG_DIR/output" \
    --bestfit "$REG_DIR/bestfit" \
    --config "./examples/canonical/local_optimizer_config.json" \
    --initial "./examples/canonical/initial.csv" ; then
    :
else
    die "Python quit unexpectedly!"
fi

python3 "$REG_DIR/compare.py" "$REG_DIR/expected_lineage.csv" "$REG_DIR/output/lineage.csv" || die "compare failed"

echo "Done testing simulated annealing."
