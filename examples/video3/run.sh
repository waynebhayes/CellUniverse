dir=$(pwd)
cd ../..
if python "./src/main.py" \
    --frame_first 22 \
    --frame_last 220 \
    --input "$dir/input/%d.jpg" \
    --bestfit "$dir/output/bestfit" \
    --output "$dir/output" \
    --config "$dir/config.json" \
    --initial "$dir/top_left.csv"  \
    --continue_from 22 \
    -ta 0 \
    -ts 1 \
    -te 0.000001 \
    --no_parallel --graySynthetic --global_optimization; then
    :
else
    die "Python quit unexpectedly!"
fi