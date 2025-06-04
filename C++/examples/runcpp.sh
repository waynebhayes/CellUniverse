dir=$(pwd)
if  ../build/celluniverse 1 19 $dir/input/frame%03d.tif $dir/output $dir/config.yaml $dir/initial.csv
    then
    :
else
    echo "Cell Universe quit unexpectedly!"
fi
