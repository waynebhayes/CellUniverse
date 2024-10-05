## Cell Labeling Tool 

A tool to label cells by mouse in images and record the labeled cells' information.

### Major Features:

1. File operations

   1. Open file: select and open new working image

   2. Save cell data (csv file):

      ​	save labeled cells' information to the csv file, including its name (by binary numbers), center coordinates, width, length, rotation (in radians).

   3. Save current cell image:

      	1. Save the current image with bounding box as jpg or png file
       	2. **Need to rescale your monitor's display setting to 100% to use this feature**

2. Bounding box operations

   1. New bounding box:

      1. Draw bounding box with dynamic display in the working image

      2. Procedure:

         1. **Click New bounding box button if not in this mode**

         2. draw a line along the longest side of the cell 

            **(1) left click mouse** 

            **(2) moving with left click pressing down** 

            **(3) release left click**

         3. then drag the bounding box to another side of the cell 

            **(1) in the previous release place, right click mouse **

            **(2) moving with right click pressing down **

            **(3) release right click **

   2. Delete bounding box:

      1.  Select the bounding box in the image and delete
      2. Procedure:
         1. **Click Delete bounding box button if not in this mode**
         2. **Left click the location inside the bounding box that you want to delete**
         3. **Current bounding box in the left side bar will show the information of your selected one**
         4. **Click ok or cancel in the pop-up window to finally delete this bounding box or cancel deleting**

3. Left Side Toolbar

   1. Current mouse location: display current mouse coordinates

   2. Current bounding box: 

      ​	Show the center coordinates, width, length, and rotation of the bounding box, which is corresponding to the cell inside it and will be export into the csv file later. 

------

### Notes:

1. The current window is set to "1280x640", and 125% - 150% for your monitor's display setting will work well on the "input" and "input_PCImg" images.
2. Due to the implementation of PIL library, the feature "Save current cell image" needs 100% for your monitor's display setting.
3. Cannot draw the bounding box with its center outside of the image boundary. 
4. In the "Delete bounding box" mode, you can also delete anything incorrectly drawn before, including lines, plots, etc. 