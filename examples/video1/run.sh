dir=$(pwd)
cd ../..
if python "./src/main.py" \
    --frame_first 0 \
    --frame_last 350 \
    --input "$dir/input/%d.png" \
    --bestfit "$dir/output/bestfit" \
    --output "$dir/output" \
    --config "$dir/config.json" \
    --initial "$dir/initial.csv"  \
    --no_parallel --graySynthetic --global_optimization; then
    :
else
    die "Python quit unexpectedly!"
fi
