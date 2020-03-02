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
#NOTE: The program will ignore comments

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

Images must all be placed in a directory. The default is `./frames/` directory, but you can change the name as long as you input the directory in the command line. The names of the images must match `0.png`, `1.png`, `2.png`, and so on.

Usage
-----

Command line help:

``` sourceCode
usage: main.py [-h] [-d DIRECTORY] [-s N] [-f N] [--dist] [-w WORKERS] [-j JOBS] [--keep KEEP] [--strategy STRATEGY] [--cluster CLUSTER] -i
               PATTERN -o DIRECTORY -c FILE -x FILE -t TEMP -e TEMP

optional arguments:
  -h, --help            show this help message and exit
  -d DIRECTORY, --debug DIRECTORY
                        path to the debug directory (enables debug mode)
  -s N, --start N       starting image (default: 0)
  -f N, --finish N      final image (defaults to until last image)
  --dist                use distance-based objective function
  -w WORKERS, --workers WORKERS
                        number of parallel workers (defaults to number of processors)
  -j JOBS, --jobs JOBS  number of jobs per frame (defaults to --workers/-w)
  --keep KEEP           number of top solutions kept (must be equal or less than --jobs/-j)
  --strategy STRATEGY   one of "best-wins", "worst-wins", "extreme-wins"
  --cluster CLUSTER     dask cluster address (defaults to local cluster)

required arguments:
  -i PATTERN, --input PATTERN
                        input filename pattern (e.g. "image%03d.png")
  -o DIRECTORY, --output DIRECTORY
                        path to the output directory
  -c FILE, --config FILE
                        path to the configuration file
  -x FILE, --initial FILE
                        path to the initial cell configuration
  -t TEMP, --temp TEMP  starting temperature for the simulated annealing
  -e TEMP, --endtemp TEMP
                        ending temperature for the simulated annealing
  -a AUTOTEMP --auto_temp 
                        enable auto-temperature schedule
```

Examples
--------

``` sourceCode
python "./main.py" --start 0 --finish 13 --debug "./debug" --input "./input/frame%03d.png" --output "$TEST_DIR/output" --config "./config.json" --initial "./cells.0.csv" --temp 10 --endtemp 0.01
```
