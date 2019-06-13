#!/bin/bash
die() { echo "$@" >&2; exit 1; }

echo "Testing cellviewer"

TEST_DIR=./regression-tests/cellviewer
[ -d "$TEST_DIR" ] || die "Must run from the repository's root directory!"

mkdir -p $TEST_DIR/test/
rm -f $TEST_DIR/test/*
cp -r $TEST_DIR/original_data/ $TEST_DIR/test/

mkdir -p $TEST_DIR/results/
rm -f $TEST_DIR/results/*



if python "./cellviewer/radialtree.py" \
    $TEST_DIR/test; then
    :
else
    die "Python quit unexpectedly!"
fi

python "$TEST_DIR/checkerCellViewer.py" \
    "-checkLevel=1" \
    "-wd=$TEST_DIR"

echo "Done testing cellviewer"
exit