dir=$(pwd)
cd ../..
if python "./src/global_optimize_main.py" \
    -ff 1 \
    -lf 9 \
    --input "$dir/input/frame%03d.tif" \
    --output "$dir/output" \
    --config "$dir/config.yaml" \
    --initial "$dir/initial.csv"
    then
    :
else
    die "Python quit unexpectedly!"
fi