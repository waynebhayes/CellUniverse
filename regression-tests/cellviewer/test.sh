#!/bin/bash -x
die() { echo "FATAL ERROR: $@" >&2; exit 1;
}

echo "Testing cellviewer"

[ -d "$REG_DIR" ] || die "Must run from the repository's root directory"

rm -rf $REG_DIR/test
mkdir $REG_DIR/test
cp -r $REG_DIR/original_data/* $REG_DIR/test

rm -rf $REG_DIR/results
mkdir $REG_DIR/results

python3 "./tools/cellviewer/radialtree.py" $REG_DIR/test || die "Python died on radialtree.py unexpectedly"

python3 "$REG_DIR/checkerCellViewer.py" "-checkLevel=1" "-wd=$REG_DIR" || die "Python died on checkerCellViewer.py"

awk '
    /CHECK$/{X=$1; ++TEST[X]}
    /Test.*Passed/{++PASS[X]}
    END{missed=0;
	for(i in TEST)if(PASS[i]!=2){++missed; print "Missed",i}
	exit(missed);
    }' $REG_DIR/results/results.txt
exit $?
