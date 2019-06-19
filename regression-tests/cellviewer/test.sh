#!/bin/bash
die() { echo "$@" >&2; exit 1; }

echo "Testing cellviewer"

TEST_DIR=./regression-tests/cellviewer
[ -d "$TEST_DIR" ] || die "Must run from the repository's root directory"

rm -rf $TEST_DIR/test
mkdir $TEST_DIR/test
cp -r $TEST_DIR/original_data/* $TEST_DIR/test

rm -rf $TEST_DIR/results
mkdir $TEST_DIR/results

python3 "./cellviewer/radialtree.py" $TEST_DIR/test || die "Python died on radialtree.py unexpectedly"

python3 "$TEST_DIR/checkerCellViewer.py" "-checkLevel=1" "-wd=$TEST_DIR" || die "Python died on checkerCellViewer.py"

echo "Done testing cellviewer"
