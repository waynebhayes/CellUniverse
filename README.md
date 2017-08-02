Cell Universe
=============

Tracks the movement of cells from a video source.

Initial Properties File
-----------------------

This is an example initial properties file:

``` sourceCode
# Initial property file (line commented out)
pos:x   pos:y   length  rotation
160     105     20      1.605
156     125     17      1.997
165     130     14      1.997
170     113     15      1.605
```
You can start with one or more cell properties. The rotation is specified in
radians.

Also, you may specify the names manually:

``` sourceCode
# Initial property file
name    pos:x   pos:y   length  rotation
"00"    160     105     20      1.605
"01"    156     125     17      1.997
"10"    165     130     14      1.997
"11"    170     113     15      1.605
```

Frames
------

Images must be placed in the `./frames/` directory. The names of the images
must match `0.png`, `1.png`, `2.png`, and so on.

Usage
-----

Command line help:

``` sourceCode
$ python celluniverse.py --help
usage: celluniverse.py [-h] [-v] [-s FRAME] [-p COUNT] initial

Cell-Universe Cell Tracker.

positional arguments:
  initial               initial properties file ('example.init.txt')

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -s FRAME, --start FRAME
                        start from specific frame (default: 0)
  -p COUNT, --processes COUNT
                        number of concurrent processes to run (default: 4)
```

Examples
--------

``` sourceCode
python celluniverse.py --processes 2 example1.init.txt
```