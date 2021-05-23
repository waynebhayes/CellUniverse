#!/bin/sh -x
die() { echo "FATAL ERROR: $@" >&2; exit 1;
}
npm() { echo "Not running npm, since it seems to be broken" >&2
}
echo "Testing cellviewer (pushing to website)"
npm install --prefix tools/cellviewer || die "npm failed"

rm -rf ./tools/cellviewer/src/output
rm -rf ./tools/cellviewer/src/bestfit
rm -rf node_modules/gh-pages/.cache
mkdir ./tools/cellviewer/src/output
mkdir ./tools/cellviewer/src/bestfit

python3 "src/main.py" \
    --frame_first 0 \
    --frame_last 13 \
    --debug "./debug" \
    --input "./examples/canonical/input/gray/frame%03d.png" \
    --output "./tools/cellviewer/src/output" \
    --bestfit "./tools/cellviewer/src/bestfit" \
    --config "./examples/canonical/local_optimizer_config.json" \
    --no_parallel \
    --initial "./examples/canonical/initial.csv" || die "main.py failed"

python3 "./tools/cellviewer/radialtree.py" \
    "./tools/cellviewer/src/output" || die "radialtree failed"

npm run --prefix tools/cellviewer deploy || die "2nd npm failed"

echo "Done testing cellviewer (pushed to website)"
