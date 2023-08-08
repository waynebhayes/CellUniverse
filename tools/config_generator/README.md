Config Generator Manual
===
This tool helps create configuration files for Cell Universe. It functions by comparing the cell information on two separate frames in a video.

---
## Usage

Step 1: Run the command to open the GUI.
```
python config_generator.py
```
After running this command, this window should show up.

![Select Folder](readme_assets/select_folder.png) 



Step 2: Click the "Select Folder" button and locate to the folder with video frames.

Once successed, you should see a window looks like this.

![Main Window](readme_assets/main_window.png)



Step 3: Fill in all the text entries that are surrounded by the red rectangle.

(Regarding how to fill in the text entries, please refer to the next section.)

![Main Window](readme_assets/required_parameters.png)



Step 4: Click Generate button. A window will appear, prompting you to select the output directory for the generated config.

---

## Functions of the program

1. Draw bounding boxes - After clicking the "new bounding box" button under the canvas, you are able to draw bounding boxes on that canvas. The method of operation is similar to the cell labelling tool. First, left click on one corner of the cell and move along the longer side of the cell until you reach the opposite side. Then, right click and move the cursor to adjust the width of the bounding box. Release the right mouse button if the bounding box correctly encloses the cell.

2. Delete boudning boxes - After clicking the "delete bounding box" button under the canvas, you are able to delete a bounding box on that canvas. You simply click on the bounding box that you want to delete, and then click "OK" on the confirmation window.

3. Select bounding boxes - After clicking the "select bounding box" button under the canvas, you are able to select a bounding box on that canvas.

4. Navigation through different frame - Under each canvas, there are some buttons and a text entry help you to browse different frames.

5. Synthetic Image Preview - The components related to this functionality is on the second column from right to left. After picking the color of background and cells, and with all text entry boxes filled, you can see the synthetic image by click the generate button in this colomn.

---

## How to determine/fill in each paramters

1. Max speed/spin/growth - This tool can help you measure the speed/spin/growth between two cells, but the max speed/spin/growth needs to be manually input by the user.

   To calculate the speed/spin speed and growth rate of a cell:
   1. Click the select bounding box button under the canvas.
   2. Click on one bounding box in the canvas.
   3. Repeat the previous steps for the other canvas. (The two selected cells need to be the same cell)
   4. Click the calculate button in the info panel.
   5. The speed/spin speed/growth rate now appears in the info panel.

   Manually enter the result into the data form. For each value, add some tolerance value to account for potential errors. For example, if the speed is 0.283, you should type 0.3 or maybe 0.35 for max speed just in case the cell you found is not the one that moves most quickly.

2. Cell/Background Color

    1.	Click the pick button on the right of input box.
    2.	Select a rectangular area on either of the canvas.
    3.	The average grey scale color should have automatically been placed in the input box.

3. Determine diffraction parameters.

    Use the synthetic preview section to find the optimal diffraction parameters. The goal is to create a closest synthetic cell image compared to the real image. Make sure you have finished step 4 before press the generate button in the synthetic preview section.

4. prob.split

    First, ensure that there is a significant difference in frame number of frame displayed on the two canvases. For example, the frame number on the left can be 0, and the frame number on the right can be 50 or 100. Fill in the "Cell count in the canvas" text entry box under each canvas(No bounding boxes needed). Click the "get" button. Then, the calculated split rate is in the entry box.


