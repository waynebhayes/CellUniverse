dir=$(pwd)
cd ../..
if python "./src/main.py" \
    --frame_first 0 \
    --frame_last 100 \
    --input "$dir/input/gray/frame%03d.png" \
    --bestfit "$dir/output/bestfit" \
    --output "$dir/output" \
    --config "$dir/multi_config.json" \
    --initial "$dir/initial.csv"  \
    --workers 8 --graySynthetic --global_optimization; then
    :
else
    die "Python quit unexpectedly!"
fi