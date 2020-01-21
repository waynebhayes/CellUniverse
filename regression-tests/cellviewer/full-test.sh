echo "Testing cellviewer (pushing to website)"
npm install --prefix cellviewer

rm -rf ./cellviewer/src/output/
rm -rf node_modules/gh-pages/.cache
mkdir ./cellviewer/src/output/

python3 "./main.py" \
    --start 0 \
    --finish 3 \
    --debug "./debug" \
    --input "./input/frame%03d.png" \
    --output "./cellviewer/src/output" \
    --config "./config.json" \
    --initial "./cells.0.csv" \
    --temp 10 \
    --endtemp 0.01

python3 "./cellviewer/radialtree.py" \
    "./cellviewer/src/output"

npm run --prefix cellviewer deploy

echo "Done testing cellviewer (pushed to website)"
