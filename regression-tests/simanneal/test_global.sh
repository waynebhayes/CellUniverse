#!/bin/bash
die() { echo "$@" >&2; exit 1;
}

echo "Testing simulated annealing global optimization"

[ -d "$REG_DIR" ] || die "Must run from the repository's root directory!"

# create the test output dir if it doesn't exist
mkdir -p $REG_DIR/output
rm -f $REG_DIR/output/*

if python3 "./main.py" \
    --frame_first 0 \
    --frame_last 13 \
    --debug "./debug" \
    --input "./input/frame%03d.png" \
    --output "$REG_DIR/output" \
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

python3 "$REG_DIR/compare.py" "$REG_DIR/expected_lineage.csv" "$REG_DIR/output/lineage.csv" || die "compare failed"

echo "Done testing simulated annealing global optimization."
