dir=$(pwd)
cd ../..
if python "./src/main.py" \
    --frame_first 22 \
    --frame_last 100 \
    --input "$dir/input/%d.jpg" \
    --bestfit "$dir/output/bestfit" \
    --output "$dir/output" \
    --config "$dir/config.json" \
    --initial "$dir/initial.csv"  \
    --continue_from 22 \
    --no_parallel --graySynthetic --global_optimization; then
    :
else
    die "Python quit unexpectedly!"
fi