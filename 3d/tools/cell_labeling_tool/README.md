## Cell Labeling Tool 

A tool to label 3D cells in a stacked image tiff by mouse in  and save the labeled cells' information in a CSV that can be reloaded.

### Major Features:

1. File operations

   1. Open file: select and open new stacked .tiff image of 3D cell

   2. Save cell data (csv file):

      ​save labeled cells' information to the csv file, will save the file (where .tiff was located), name (unique hash per cell), x, y, z, r, z_scaling, split_alpha, opacity

   3. load file:

      reloads the saved csv file to either fix or add additional labels

2. Bounding box operations

   1. New bounding box:

      1. Draw bounding box in the center frame (top and bottom frames help display z-levels)

      2. Procedure:

         1. **Click New bounding box button if not in this mode**

         2. draw a circle/sphere around the cell 

            **(1) left click mouse** 

            **(2) moving with left click pressing down** 

            **(3) release left click only if you want to create a single label**

            **(3) if you want to create a sphere starting at current center hold down right click and move mouse in a vertical direction**

            **(4) on the left hand side center z-level will determine where the center of the sphere will be if center z-level is equal to current z-level then a sphere will be drawn at the current center frame**


   2. Delete bounding box:

      1.  Select the bounding box in the image and delete
      2. Procedure:
         1. **Click Delete bounding box button if not in this mode**
         2. **Left click the green edge of the bounding box that you want to delete**
         3. **the cell that has a red outline will be the deleted bounding box**
         4. **Click ok or cancel in the pop-up window to finally delete this bounding box or cancel deleting**

3. Left Side Toolbar

   1. Current mouse location: display current mouse coordinates

   2. current z-level: display current z-index for tiff file (frame 0 is current z-level 0, frame 1 is current z-level 1, etc.)

   3. center z-level: display the z-level to draw the sphere at (used when drawing sphere)

   3. z-scaling: z-scaling can be changed on the left hand side to change the number of pixels between each z-frame in image (type a number into box and click 'change z-scaling')

------

### Notes: