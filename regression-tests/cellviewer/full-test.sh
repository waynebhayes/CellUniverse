#!/bin/sh
die() { echo "FATAL ERROR: $@" >&2; exit 1;
}
npm() { echo "Not running npm, since it seems to be broken" >&2
}
echo "Testing cellviewer (pushing to website)"
npm install --prefix cellviewer || die "npm failed"

rm -rf ./cellviewer/src/output
rm -rf ./cellviewer/src/bestfit
rm -rf node_modules/gh-pages/.cache
mkdir ./cellviewer/src/output
mkdir ./cellviewer/src/bestfit

python3 "src/main.py" \
    --frame_first 0 \
    --frame_last 13 \
    --debug "./debug" \
    --input "./input/frame%03d.png" \
    --output "./cellviewer/src/output" \
    --bestfit "./cellviewer/src/bestfit" \
    --config "./local_optimizer_config.json" \
    --initial "./cells.0.csv" || die "main.py failed"

python3 "tools/cellviewer/radialtree.py" \
    "./cellviewer/src/output" || die "radialtree failed"

npm run --prefix cellviewer deploy || die "2nd npm failed"

echo "Done testing cellviewer (pushed to website)"
