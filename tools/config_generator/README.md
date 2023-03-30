# Config Generator

This tool aims to help create configuration files for Cell Universe. It functions by conparing the cell information on two seperated frame in a video. (e.g. Frame 0 and Frame 20)

# Usage

## Step 1

Open the program by typing the following command in the terminal.  
```
python config_generator.py
```

## Step 2

Click the load image button on the left canvas. Select the image you want to import in the pop-up window. Remember the frame number of the image, and type it in the 'Frame# of left' text box.

## Step 3

Click the load image button on the right canvas. Select the image you want to import in the pop-up window. Remember the frame number of the image, and type it in the 'Frame# of right' text box.

## Step 4

Draw some bounding boxes on both canvas. You don't need to draw all of them, but make sure to draw
the cells that you think has highest/lowest length or width. After finishing the drawing, click the
update button on the right. The program will show the max/min cell length and width according to the
cells thar you have drawn.

## Step 5

Next, we want to find a good estimate for cell speed, spin speed and growth rate. To obtain this,
select one cell from each canvas. The choice of cell is not arbitary. You need to make sure they
are the same cell but in different frame. In addition to that, you want to find the pair of cells  
that has the greatest change in locaition/orientation/size.

To select one cell, press the selecting bouding box button and then left click one bounding box in  
the image.

After selecting both cells, click the calculate button on the right, it will show the speed/spin speed and growth rate for this pair of cells.

## Step 6

Type the speed/spin speed/growth rate obtained in Step 5 into the text boxes below. Remember to mannualy rise the data obained by a little bit, since it's possible that the data you collect is
not the maximum.

Tips: You may want to run Step 5 on different pair of cells to find the maximum data.

## Step 7
Click generate button. Then specify the output folder for generated config file.

# TODO:

1. Currently the max/min length/width of config file is set to max/min values of bouding boxes. It's possible there are smaller or larger cells in other frames.  
2. The layout should be improved.  
3. New functionality - Select background color
4. New functionality - Mearsure diffraction light
5. New functionality - More....