dir=$(pwd)
cd ..
if  ../C++/build/celluniverse 1 9 $dir/input/frame%03d.tif $dir/output $dir/config.yaml $dir/initial.csv
    then
    :
else
    echo "Cell Universe quit unexpectedly!"
fi
