dir=$(pwd)
cd ../..
if python "./src/main.py" \
    --frame_first 0 \
    --frame_last 30 \
    --input "$dir/input/gray/frame%03d.png" \
    --bestfit "$dir/output/bestfit" \
    --output "$dir/output" \
    --config "$dir/global_optimizer_config.json" \
    --initial "$dir/initial.csv"  \
    --no_parallel --graySynthetic --global_optimization; then
    :
else
    die "Python quit unexpectedly!"
fi