#!/bin/bash -x
die() { echo "$@" >&2; exit 1;
}

echo "Testing 2D Gradient Descent Prototype"

[ -d "$REG_DIR" ] || die "Must run from the repository's root directory!"

# create the test output dir if it doesn't exist
rm -rf $REG_DIR/output
mkdir  $REG_DIR/output

rm -rf $REG_DIR/bestfit
mkdir $REG_DIR/bestfit

dir="./gradient_descent_prototype/video0"

if python "./gradient_descent_prototype/optimizationGD.py" \
    --frame_first 0 \
    --frame_last 10 \
    --input "$dir/input/gray/frame%03d.png" \
    --bestfit "$REG_DIR/bestfit" \
    --output "$REG_DIR/output" \
    --config "$dir/global_optimizer_config.json" \
    --initial "$dir/initial.csv"  \
    --no_parallel --graySynthetic --global_optimization; then
    :
else
    die "Python quit unexpectedly!"
fi

python3 "$REG_DIR/compare.py" "$REG_DIR/output/lineage.csv" "$REG_DIR/expected_lineage.csv" || die "compare failed"

echo "Done testing 2D gradient descent."
