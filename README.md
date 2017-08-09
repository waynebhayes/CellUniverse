Cell Universe
=============

Tracks the movement of cells from a video source.

Initial Properties File
-----------------------

This is an example initial properties file:

``` sourceCode
# Initial property file (line commented out)

0.33   	  	   	#dt, or the number of hours per frame
26    			#init_length of bacteria
6     			#init_width of bacteria
3     			#max speed
0.3141592653589793	#max spin
20			#number of ideal universes
3			#max x motion
3			#max y motion
7			#max x resolution
7			#max y resolution
0.3141592653589793	#max rotation
21			#max rotation resolution
0			#minimum height increase
3			#maximum height increase
4			#height increase resolution
31			#maximum length of single bacteria before splitting
13			#minimum length of any bacteria
0.25			#beginning of split ratio
0.75			#end of split ratio
20			#split ratio resolution
#NOTE: PLEASE DON'T ADD THE COMMENTS WITH THE CONFIG VARIABLES ABOVE
#OR ELSE THE CODE WON'T WORK
#SEE THE SECOND EXAMPLE FOR CLARIFICATION

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

0.33
26
6
3
0.3141592653589793
20
3
3
7
7
0.3141592653589793
21
0
3
4
31
13
0.25
0.75
20

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