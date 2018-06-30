Cell Universe
=============

Tracks the movement of cells from a video source.

Initial Properties File
-----------------------

This is an example initial properties file:

``` sourceCode
cellType: bacilli
timestep:   1.0   # second
maxSpeed:  12.0   # microns per second
maxSpin:    1.57  # radians per second
minGrowth: -2.0   # microns
maxGrowth:  9.0   # microns
minWidth:   3.0   # microns
maxWidth:   9.0   # microns
minLength: 10.0   # microns
maxLength: 48.0   # microns
delimiter: " "
---
name      x     y   width  length  rotation
A       160   105       6      18      1.61
B       156   125       6      17      2.00
C       165   130       6      16      2.00
D       170   113       6      16      1.61
```

You can start with one or more cell properties. The rotation is specified in
radians.

Frames
------

Images must all be placed in a directory. The images are found using a pattern
in the command line. For example, for the images "./input/frame001.png",
"./input/frame002.png", ..., you would use "-i frame%03d.png" to find the images.

Usage
-----

Command line help:

``` sourceCode
$ python main.py --help
usage: main.py [-h] [-v] [-q] [-l FILE] [-p N] [-s N] [-f N] -i PATTERN -o
               DIRECTORY -c FILE

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         give more output (additive)
  -q, --quiet           give less output (additive)
  -l FILE, --log FILE   path to a verbose appending log
  -p N, --processes N   number of extra processes allowed
  -s N, --start N       starting image number (default: 0)
  -f N, --finish N      final image number (default until last image

required arguments:
  -i PATTERN, --input PATTERN
                        input filename pattern (example: "image%03d.png")
  -o DIRECTORY, --output DIRECTORY
                        path to the output directory
  -c FILE, --config FILE
                        filename of the configuration file
```

Examples
--------

``` sourceCode
python main.py -s 0 -f 19 -i ./input/frame%03d.png -o ./output/ -c ./config.yml
```
