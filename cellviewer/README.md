# Cell Viewer
User Interface for Cell Universe

### Usage
#### Using Sample Data
For only viewing, can open at https://metanovitia.github.io/cellviewer/#/.

For local copy:
- Requirements: npm
- Download/clone files from simmaneal folder
- cd into cellviewer
- Run ```npm install```
- Use ```npm start``` and open localhost:3000

#### Using Your Own Data
- Requirements: Python3.7, svgwrite (installable from pip)
- Download/clone files from simmaneal folder
- Run simmaneal main.py code (Check Simmaneal Documentation for more info)
- Move the output folder into cellviewer
- cd into cellviewer
- Run ```python3.7 radialtree.py [output]``` where ```[output]``` is your output directory that you just moved, without brackets
- Drag and drop your ```[output]``` folder to the file dropzone on the website


### Assumptions
- simmaneal produce an output folder that contains at least:
  - colony.csv 
    - with headers: frame,name,x,y,width,length,rotation
    - sorted in ascending order according to frame number
    - each cell's name starts with a 'b'
    - each cell's children (when it splits) is name+'0' and name+'1'
  - images for each frame
    - same dimensions

### Limitations
- Colors are limited (4)
- Not tested with other versions of Python
- Browsers tested : Chrome, Safari, FireFox
- Browsers failed : Edge
- Bad mobile compatibility
- Lag in more dense frames

### To Do
- Add color picker
- Raw Image Files
- Optimize
  - Original Image Size should be obtained quicker, assume same dimension for each frame
- Pipeline from simmanneal