from tkinter import ttk
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog, messagebox
import tkinter.simpledialog as simpledialog
from PIL import Image, ImageTk, ImageGrab
from point import Point

import numpy as np
import platform
from collections import defaultdict
from enum import Enum

def is_point_inside_polygon(point, polygon):
        # point: a tuple of (x, y) coordinates of the point
        # polygon: a list of tuples of (x, y) coordinates of the vertices of the polygon

        # Calculate the area of the quadrilateral formed by the four points
        x1, y1 = polygon[0]
        x2, y2 = polygon[1]
        x3, y3 = polygon[2]
        x4, y4 = polygon[3]
        area_quad = abs(x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1 - x2 * y1 - x3 * y2 - x4 * y3 - x1 * y4) / 2

        # Divide the quadrilateral into two triangles by drawing a diagonal
        triangle1 = [point, polygon[0], polygon[2]]
        triangle2 = [point, polygon[1], polygon[3]]

        # Calculate the area of each triangle formed by the point and two adjacent vertices of the quadrilateral
        area_tri1 = abs(triangle1[0][0] * triangle1[1][1] + triangle1[1][0] * triangle1[2][1] + triangle1[2][0] * triangle1[0][1]
                        - triangle1[1][0] * triangle1[0][1] - triangle1[2][0] * triangle1[1][1] - triangle1[0][0] * triangle1[2][1]) / 2

        area_tri2 = abs(triangle2[0][0] * triangle2[1][1] + triangle2[1][0] * triangle2[2][1] + triangle2[2][0] * triangle2[0][1]
                        - triangle2[1][0] * triangle2[0][1] - triangle2[2][0] * triangle2[1][1] - triangle2[0][0] * triangle2[2][1]) / 2

        # If the sum of the areas of the two triangles is equal to the area of the quadrilateral, then the point is inside the polygon
        if area_tri1 + area_tri2 == area_quad:
            return True
        else:
            return False
def rotation_angle(p0, p1):
    # compute rotation angle
    x = p1.x - p0.x
    y = p1.y - p0.y
    # print(x)
    angle = np.degrees(np.arctan2(y, x))
    if angle < 0:
        angle = 360 + angle
    angle = np.radians(angle)
    return angle

def compute_edge_length(p0, p1):
    x = abs(p1.x - p0.x)
    y = abs(p1.y - p0.y)
    length = np.sqrt(x**2 + y**2)
    return length

def map_p2(p0, p1, p2):
    """
    Given three points `p0`, `p1`, and `p2`, maps the point `p2` to a new point `mapped_p2` that is on the line passing through `p2` 
    which parellels to the line passing through `p0` and `p1`, such that `mapped_p2` is the closest point to the line `p0``p1`.
    Returns the `mapped_p2` point.

    Args:
        p0 (Point): The first point on the line.
        p1 (Point): The second point on the line.
        p2 (Point): The point to be mapped.

    Returns:
        mapped_p2 (Point): The new location of p2
    """
    def is_clockwise(p1, p2, p3):
        """
        Returns True if the three points p1, p2, and p3 are in clockwise order,
        and False otherwise.
        """
        # Compute the cross product of the vectors formed by the points
        cross_product = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])

        # Check the sign of the cross product
        if cross_product < 0:
            return True
        else:
            return False
    def line_equation(x1, y1, x2, y2):
        """
        Returns the equation of the line passing through the two points (x1, y1) and (x2, y2)
        in the form ax + by + c = 0.
        """
        if x2 == x1:
            return (1, 0, -x1)  # special case for vertical line
        m = (y2 - y1) / (x2 - x1)
        a = m
        b = -1
        c = -m * x1 + y1
        return (a, b, c)
    a, b, c = line_equation(*p0, *p1)

    # the distance of a point (m, n) to the line ax + by +c =0
    # is：|ma + nb + c| / sqrt(a^2 + b^2)
    distance = (abs(p2.x * a + p2.y * b + c)) / (np.sqrt(a ** 2 + b ** 2))
    
    # determine the orientation of p2
    p2_is_clockwise = is_clockwise(p0, p1, p2)

    # get the perpendicular vector
    if(p2_is_clockwise):
        perpendicular_vec = ((p1.y - p0.y), -(p1.x - p0.x))
    else:
        perpendicular_vec = (-(p1.y - p0.y), (p1.x - p0.x))


    # normalize it to unit vector
    magnitude = np.sqrt(perpendicular_vec[0] ** 2 + perpendicular_vec[1] ** 2)

    perpendicular_vec = (perpendicular_vec[0] / magnitude, perpendicular_vec[1] / magnitude)

    mapped_p2 = Point([int(p1.x + distance * perpendicular_vec[0]),int(p1.y + distance * perpendicular_vec[1])])

    # print(f'Point 0: {p0}, Point 1:{p1}, Point 2:{p2}')
    # print(f'equation: {a}x+{b}y+{c}=0')
    # print(f'is_clockwise: {p2_is_clockwise}')
    # print(f'distance: {distance}')
    # print(f'perpendicular_vec: {perpendicular_vec}')
    return mapped_p2

def rgb_to_grayscale(r, g, b):
    return 0.2989 * r + 0.5870 * g + 0.1140 * b 
class ImageCanvasFrame(Frame):
    KWARGS = {
        "padding": 0
    }
    def __init__(self, root, idx, **kwargs):
        # Merge default kwargs with user-supplied kwargs
        self.merged_kwargs = {**self.KWARGS, **kwargs}
        super().__init__(root, **self.merged_kwargs)

        # idx indicates which ImageCanvasFrame it is.
        # 1 for the left frame, 2 for the right frame
        self.idx = idx

        self.imageCanvas = ImageCanvas(self)
        self.imageCanvas.pack()

        # Add a reset button to the frame
        self.reset_button = Button(self, text="Reset", command=self.reset_canvas, state='disabled')
        self.reset_button.pack()

        # Add a new bounding box button to the frame
        self.new_bounding_box_button = Button(self, text="New Bounding Box", command=self.new_bounding_box, width=20)
        self.new_bounding_box_button.pack()

        # Add a delete bounding box button to the frame
        self.delete_bounding_box_button = Button(self, text="Delete Bounding Box", command=self.delete_bounding_box, width=20)
        self.delete_bounding_box_button.pack()

        # Add a select bounding box button to the frame
        self.select_bounding_box_button = Button(self, text="Select Bounding Box", command=self.select_bounding_box, width=20)
        self.select_bounding_box_button.pack()

        # Add a mouse position frame to the frame
        self.mouse_position_frame = MousePositionFrame(self)
        self.mouse_position_frame.pack()

        # Add a bounding box frame to the frame
        self.bouding_box_info_frame = BoundingBoxInfoFrame(self)
        self.bouding_box_info_frame.pack()

    # Reset Button Callbacks
    def reset_canvas(self):
        # Reset the canvas to its original state
        self.imageCanvas.delete("all")
        self.imageCanvas.reset_var()

        # Recover the button states
        self.new_bounding_box_button.configure(state='normal')
        self.delete_bounding_box_button.configure(state='normal')
        self.select_bounding_box_button.configure(state='normal')
        self.reset_button.configure(state='disabled')

        # Recover the flag variables
        self.imageCanvas.image_loaded = False
        self.imageCanvas.status = ImageCanvasStatus.WAIT_FOR_LOADING_IMAGE
        
        # Clear the frame number oin the input panel
        if (self.idx == 1):
            self.master.input_panel.frame_number1.delete(0, END)
        elif(self.idx == 2):
            self.master.input_panel.frame_number2.delete(0, END)

        # Set the canvas to original size
        self.imageCanvas.config(width=self.imageCanvas.canvas_width, height=self.imageCanvas.canvas_height)

        # Reconstruct the buttons
        self.imageCanvas.load_button_id = self.imageCanvas.create_window(self.imageCanvas.canvas_width / 2, self.imageCanvas.canvas_height / 2, anchor="center", window=self.imageCanvas.load_button)
        self.imageCanvas.frame_input_label_id = self.imageCanvas.create_window(self.imageCanvas.canvas_width / 2, self.imageCanvas.canvas_height / 2 - 25, anchor="center", window=self.imageCanvas.frame_input_label)
        self.imageCanvas.frame_input_box_id = self.imageCanvas.create_window(self.imageCanvas.canvas_width / 2, self.imageCanvas.canvas_height / 2, anchor="center", window=self.imageCanvas.frame_input_box)
        self.imageCanvas.frame_confirm_button_id = self.imageCanvas.create_window(self.imageCanvas.canvas_width / 2, self.imageCanvas.canvas_height / 2 + 25, anchor="center", window=self.imageCanvas.frame_confirm_button)
        
        # Set the windows that related to frame number input to hidden
        self.imageCanvas.itemconfigure(self.imageCanvas.frame_input_label_id, state='hidden')
        self.imageCanvas.itemconfigure(self.imageCanvas.frame_input_box_id, state='hidden')
        self.imageCanvas.itemconfigure(self.imageCanvas.frame_confirm_button_id, state='hidden')

        # Clear the previously drawn cells
        self.imageCanvas.cell_dict.clear()


    # New Bounding Box Button Callback
    def new_bounding_box(self):
        if(self.imageCanvas.status == ImageCanvasStatus.WAIT_FOR_LOADING_IMAGE or
           self.imageCanvas.status == ImageCanvasStatus.WAIT_FOR_FRAME_NUMBER):
            messagebox.showerror("Error", "Please first load the image and provide the frame number.")
            return
        self.imageCanvas.status = ImageCanvasStatus.DRAWING
        self.new_bounding_box_button.configure(state='disable')
        self.delete_bounding_box_button.configure(state='normal')
        self.select_bounding_box_button.configure(state='normal')


    # Delete Bounding Box Button Callback
    def delete_bounding_box(self):
        if(self.imageCanvas.status == ImageCanvasStatus.WAIT_FOR_LOADING_IMAGE or
           self.imageCanvas.status == ImageCanvasStatus.WAIT_FOR_FRAME_NUMBER):
            messagebox.showerror("Error", "Please first load the image and provide the frame number.")
            return
        self.imageCanvas.status = ImageCanvasStatus.DELETING
        self.new_bounding_box_button.configure(state='normal')
        self.delete_bounding_box_button.configure(state='disable')
        self.select_bounding_box_button.configure(state='normal')


    # Select Bounding Box Button Callback
    def select_bounding_box(self):
        if(self.imageCanvas.status == ImageCanvasStatus.WAIT_FOR_LOADING_IMAGE or
           self.imageCanvas.status == ImageCanvasStatus.WAIT_FOR_FRAME_NUMBER):
            messagebox.showerror("Error", "Please first load the image and provide the frame number.")
            return
        self.imageCanvas.status = ImageCanvasStatus.SELECTING
        self.new_bounding_box_button.configure(state='normal')
        self.delete_bounding_box_button.configure(state='normal')
        self.select_bounding_box_button.configure(state='disable')


    # Enter pick cell color mode
    # This function is called when the pick button in the input panel is pressed
    def pick_cell_color(self):
        if(self.imageCanvas.status == ImageCanvasStatus.WAIT_FOR_LOADING_IMAGE or
           self.imageCanvas.status == ImageCanvasStatus.WAIT_FOR_FRAME_NUMBER):
            return False
        self.imageCanvas.status = ImageCanvasStatus.PICKING_CELL_COLOR
        self.new_bounding_box_button.configure(state='normal')
        self.delete_bounding_box_button.configure(state='normal')
        self.select_bounding_box_button.configure(state='normal')
        return True

    # Enter pick background color mode
    # This function is called when the pick button in the input panel is pressed
    def pick_background_color(self):
        if(self.imageCanvas.status == ImageCanvasStatus.WAIT_FOR_LOADING_IMAGE or
           self.imageCanvas.status == ImageCanvasStatus.WAIT_FOR_FRAME_NUMBER):
            return False
        self.imageCanvas.status = ImageCanvasStatus.PICKING_BACKGROUND_COLOR
        self.new_bounding_box_button.configure(state='normal')
        self.delete_bounding_box_button.configure(state='normal')
        self.select_bounding_box_button.configure(state='normal')
        return True

    # Enter Idle mode
    # This function is called after picking a color
    def idle_mode(self):
        self.imageCanvas.status = ImageCanvasStatus.IDLE
        self.new_bounding_box_button.configure(state='normal')
        self.delete_bounding_box_button.configure(state='normal')
        self.select_bounding_box_button.configure(state='normal')
class ImageCanvasStatus(Enum):
    WAIT_FOR_LOADING_IMAGE = 1
    WAIT_FOR_FRAME_NUMBER = 2
    IDLE = 3
    DRAWING = 4
    DELETING = 5
    SELECTING = 6
    PICKING_CELL_COLOR = 7
    PICKING_BACKGROUND_COLOR = 8

class ImageCanvas(Canvas):
    KWARGS = {
        "bg": "white",
        "width": 400,
        "height": 400,
    }
    def __init__(self, root, **kwargs):
        # Merge default kwargs with user-supplied kwargs
        self.merged_kwargs = {**self.KWARGS, **kwargs}
        super().__init__(root, **self.merged_kwargs)

        self.image_loaded = False
        self.status = ImageCanvasStatus.WAIT_FOR_LOADING_IMAGE
        self.zoom_factor = None    # Raw Image Width / Canvas Width

        self.selected_id = None
        # Variables describing a bouding box
        self.temp_item = None      # Incomplete bounding box
        self.p0 = None             # p0 is the cursor location when LB is pressed
        self.p1 = None             # p1 is the cursor location when LB is stopped/released
        self.p2 = None             # p2 is the cursor location when RB is pressed
        self.p3 = None             # p3 is calculated according to previous three points
        self.angle = None
        self.width = None
        self.length = None
        self.center = None
        self.current_id = None

        self.cell_dict = defaultdict(dict)
        self.canvas_width = self.merged_kwargs["width"]
        self.canvas_height = self.merged_kwargs["height"]
        self.plt = platform.system()

        # Defines the load image button and the related widgets for typing the frame number
        self.load_button = Button(self, text="Load Image", command=self.load_image)
        self.frame_input_label = Label(self, text="Enter the frame number.")
        self.frame_input_box = Entry(self)
        self.frame_confirm_button = Button(self, text="Confirm", command=self.confirm_frame_number)

        self.load_button_id = self.create_window(self.canvas_width / 2, self.canvas_height / 2, anchor="center", window=self.load_button)
        self.frame_input_label_id = self.create_window(self.canvas_width / 2, self.canvas_height / 2 - 25, anchor="center", window=self.frame_input_label)
        self.frame_input_box_id = self.create_window(self.canvas_width / 2, self.canvas_height / 2, anchor="center", window=self.frame_input_box)
        self.frame_confirm_button_id = self.create_window(self.canvas_width / 2, self.canvas_height / 2 + 25, anchor="center", window=self.frame_confirm_button)

        # Set the windows that related to frame number input to hidden
        self.itemconfigure(self.frame_input_label_id, state='hidden')
        self.itemconfigure(self.frame_input_box_id, state='hidden')
        self.itemconfigure(self.frame_confirm_button_id, state='hidden')

    # Define a function to handle button clicks
    def load_image(self):
        # Prompt the user for an image file and its index
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        self.image = Image.open(file_path)

        # Calculate the aspect ratio of the image and the canvas
        image_width, image_height = self.image.size
        canvas_aspect_ratio = self.canvas_width / self.canvas_height
        image_aspect_ratio = image_width / image_height

        # If the image is wider than the canvas, resize it to fit the width of the canvas
        if image_aspect_ratio > canvas_aspect_ratio:
            self.new_canvas_width = self.canvas_width
            self.new_canvas_height = int(self.new_canvas_width / image_aspect_ratio)

        # If the image is taller than the canvas, resize it to fit the height of the canvas
        else:
            self.new_canvas_height = self.canvas_height
            self.new_canvas_width = int(self.new_canvas_height * image_aspect_ratio)

        # Resize the image and display it on the canvas
        self.image_resized = self.image.resize((self.new_canvas_width, self.new_canvas_height))
        self.config(width=self.new_canvas_width, height=self.new_canvas_height)

        self.image_tk = ImageTk.PhotoImage(self.image_resized)
        self.image_id = self.create_image(0, 0, anchor="nw", image=self.image_tk)

        self.image_loaded = True
        self.zoom_factor = image_width / self.canvas_width
        # Hide the load image button on the canvas
        self.itemconfigure(self.load_button_id, state='hidden')

        # Then prompt the user for the frame number
        # Set the related windows/wigets to normal
        self.itemconfigure(self.frame_input_label_id, state='normal')
        self.itemconfigure(self.frame_input_box_id, state='normal')
        self.itemconfigure(self.frame_confirm_button_id, state='normal')

        self.status = ImageCanvasStatus.WAIT_FOR_FRAME_NUMBER

    def confirm_frame_number(self):
        frame_number = self.frame_input_box.get()
        try:
            int(frame_number)
        except ValueError:
            messagebox.showerror("Error", "Please provide a valid integer.")
        
        # Also change the value in the input panel
        if(self.master.idx == 1):
            frame_number_entry = self.master.master.input_panel.frame_number1
        elif(self.master.idx == 2):
            frame_number_entry = self.master.master.input_panel.frame_number2
        else:
            raise Exception("The index of image canvas should be only 1 or 2")
        
        frame_number_entry.delete(0, END)
        frame_number_entry.insert(0, frame_number)

        # Set the related windows/wigets to normal
        self.itemconfigure(self.frame_input_label_id, state='hidden')
        self.itemconfigure(self.frame_input_box_id, state='hidden')
        self.itemconfigure(self.frame_confirm_button_id, state='hidden')

        # Activate the reset button
        self.status = ImageCanvasStatus.IDLE
        self.master.reset_button.config(state='normal')
        self.bind('<ButtonPress-1>', self.left_press_down)
        self.bind('<Motion>', self.mouse_move)
        
    ####################################################
    #      Callback functions for canvas
    ####################################################
    def calculate_vertices2(self, p0, p1, p2):
        self.angle = rotation_angle(p0, p1)
        self.p2 = map_p2(p0, p1, p2)
        p2 = self.p2
        self.width = compute_edge_length(p1, p2)
        self.length = compute_edge_length(p0, p1)
        self.center = ((p0.x + p2.x) / 2, (p0.y + p2.y) / 2)
        self.p3 = Point([(p0.x + p2.x - p1.x), (p0.y + p2.y - p1.y)])

    def left_press_down(self, event):
        if self.temp_item is not None:
            self.delete(self.temp_item)

        if self.status == ImageCanvasStatus.DRAWING:
            self.p0 = Point([self.canvasx(event.x),
                            self.canvasy(event.y)])
            self.bind('<B1-Motion>', self.left_motion)
        
        elif self.status == ImageCanvasStatus.SELECTING:
            mx = self.canvasx(event.x)
            my = self.canvasy(event.y)
            # get canvas object ID of where mouse pointer is
            canvasobject = self.find_closest(mx, my, halo=5)

            if canvasobject == () or self.image_id == canvasobject[0]:
                return
            if(self.selected_id == None):
                self.selected_id = canvasobject[0]
                self.update_cell_label(self.selected_id)
                self.itemconfigure(self.selected_id, outline='red')
            elif self.selected_id == canvasobject[0]:
                self.itemconfigure(self.selected_id, outline='green')
                self.selected_id = None
                self.clear_cell_label()
            else:
                self.itemconfigure(self.selected_id, outline='green')
                self.selected_id = canvasobject[0]
                self.update_cell_label(self.selected_id)
                self.itemconfigure(self.selected_id, outline='red')
                
            
        elif self.status == ImageCanvasStatus.DELETING:
            mx = self.canvasx(event.x)
            my = self.canvasy(event.y)
            # get canvas object ID of where mouse pointer is
            canvasobject = self.find_closest(mx, my, halo=5)

            # print(canvasobject)
            # self.delete(canvasobject)

            if canvasobject == () or self.image_id == canvasobject[0]:
                return
            
            # first change the selected bounding box
            if(self.selected_id == None):
                self.selected_id = canvasobject[0]
                self.update_cell_label(self.selected_id)
                self.itemconfigure(self.selected_id, outline='red')
            elif self.selected_id == canvasobject[0]:
                # if the box to be deleted is the same, don't deselect it.
                pass
            else:
                self.itemconfigure(self.selected_id, outline='green')
                self.selected_id = canvasobject[0]
                self.update_cell_label(self.selected_id)
                self.itemconfigure(self.selected_id, outline='red')

            delete = messagebox.askokcancel('Warning', 'Delete this bounding box?')
            if delete is True:
                self.delete(canvasobject[0])
                del self.cell_dict[canvasobject[0]]
                self.selected_id = None
                self.clear_cell_label()
                #print("delete: {}".format(canvasobject[0]))
                # print(self.cell_dict)
        elif self.status == ImageCanvasStatus.PICKING_CELL_COLOR or self.status == ImageCanvasStatus.PICKING_BACKGROUND_COLOR:
            self.p0 = Point([self.canvasx(event.x),
                            self.canvasy(event.y)])
            self.bind('<B1-Motion>', self.left_motion)

    def left_motion(self, event):
        if self.status == ImageCanvasStatus.DRAWING:
            if self.temp_item is not None:
                self.delete(self.temp_item)
            self.p1 = Point([self.canvasx(event.x),
                            self.canvasy(event.y)])
            self.temp_item = self.create_line(
                *self.p0, *self.p1, fill='red', width=3)
            self.bind('<ButtonRelease-1>', self.stop_left_move)
        elif self.status == ImageCanvasStatus.PICKING_CELL_COLOR or self.status == ImageCanvasStatus.PICKING_BACKGROUND_COLOR:
            if self.temp_item is not None:
                self.delete(self.temp_item)
            self.p1 = Point([self.canvasx(event.x),
                            self.canvasy(event.y)])
            self.temp_item = self.create_rectangle(
                *self.p0, *self.p1, outline='red', width=3)
            self.bind('<ButtonRelease-1>', self.stop_left_move)
    def stop_left_move(self, event):
        if self.status == ImageCanvasStatus.DRAWING:
            if self.temp_item is not None:
                self.delete(self.temp_item)

            self.p1 = Point([self.canvasx(event.x),
                            self.canvasy(event.y)])

            # print(self.p0)
            # print(self.p1)

            self.temp_item = self.create_line(
                *self.p0, *self.p1, fill='red', width=3)
            if self.plt == "Darwin":
                # 2 for MacOS Right-Click, 3 for Windows
                self.bind('<ButtonPress-2>', self.start_right_move)
            else:
                self.bind('<ButtonPress-3>', self.start_right_move)
        elif self.status == ImageCanvasStatus.PICKING_CELL_COLOR or self.status == ImageCanvasStatus.PICKING_BACKGROUND_COLOR:
            if self.temp_item is not None:
                self.delete(self.temp_item)
            self.p1 = Point([self.canvasx(event.x),
                            self.canvasy(event.y)])
            self.temp_item = self.create_rectangle(
                *self.p0, *self.p1, outline='red', width=3)
            self.delete(self.temp_item)

            # calculate the raw coordinates
            raw_p0 = self.p0 * self.zoom_factor
            raw_p1 = self.p1 * self.zoom_factor

            # get the pixel colors from the raw image
            pixel_colors = []
            for x in range(int(raw_p0.x), int(raw_p1.x)):
                for y in range(int(raw_p0.y), int(raw_p1.y)):
                    pixel_colors.append(self.image.getpixel((x, y)))
            avg_color = tuple(np.mean(pixel_colors, axis=0, dtype=np.uint32))
            gray_scale_color = rgb_to_grayscale(*avg_color) / 255

            if(self.status == ImageCanvasStatus.PICKING_BACKGROUND_COLOR):
                background_color_entry = self.master.master.input_panel.background_color
                background_color_entry.delete(0, END)
                background_color_entry.insert(0, f'{gray_scale_color:.3f}')

                color_info_panel = self.master.master.informationPanel.colorInfoPanel
                color_info_panel.set_background_color(gray_scale_color)
            elif(self.status == ImageCanvasStatus.PICKING_CELL_COLOR):
                cell_color_entry = self.master.master.input_panel.cell_color
                cell_color_entry.delete(0, END)
                cell_color_entry.insert(0, f'{gray_scale_color:.3f}')

                color_info_panel = self.master.master.informationPanel.colorInfoPanel
                color_info_panel.set_cell_color(gray_scale_color)
            
            self.master.idle_mode()
            self.reset_var()
    def start_right_move(self, event):
        if self.status == ImageCanvasStatus.DRAWING:
            self.p2 = Point([self.canvasx(event.x),
                            self.canvasy(event.y)])
            if self.plt == "Darwin":
                self.bind('<B2-Motion>', self.right_motion)
            else:
                self.bind('<B3-Motion>', self.right_motion)
    
    def right_motion(self, event):
        if self.status == ImageCanvasStatus.DRAWING:
            if self.temp_item is not None:
                self.delete(self.temp_item)

            self.p2 = Point([self.canvasx(event.x),
                            self.canvasy(event.y)])

            self.calculate_vertices2(self.p0, self.p1, self.p2)

            self.temp_item = self.create_polygon(self.p0[0], self.p0[1],
                                                        self.p1[0], self.p1[1],
                                                        self.p2[0], self.p2[1],
                                                        self.p3[0], self.p3[1],
                                                        fill='', outline='red', width=3)
            if self.plt == "Darwin":
                self.bind('<ButtonRelease-2>', self.stop_right_move)
            else:
                self.bind('<ButtonRelease-3>', self.stop_right_move)

    def stop_right_move(self, event):
        if self.status == ImageCanvasStatus.DRAWING:
            if self.temp_item is not None:
                self.delete(self.temp_item)

            self.p2 = Point([self.canvasx(event.x),
                            self.canvasy(event.y)])

            # print(self.p0)

            self.calculate_vertices2(self.p0, self.p1, self.p2)
            self.current_id = self.create_polygon(self.p0[0], self.p0[1],
                                                        self.p1[0], self.p1[1],
                                                        self.p2[0], self.p2[1],
                                                        self.p3[0], self.p3[1],
                                                        fill='', outline='red', width=3)
            # print(self.current_id)

            # check the center inside the image
            if self.center[0] > self.new_canvas_width or self.center[0] < 0 or \
                    self.center[1] > self.new_canvas_height or self.center[1] < 0:
                messagebox.showerror(
                'Error', 'The center of the bounding box is outside of the image! Please draw again.')
                self.delete(self.current_id)
            else:
                # print("add:{}".format(self.current_id))
                # save to cell_dict
                self.save_cell(self.current_id, self.center, self.width, self.length, self.angle)
                self.update_cell_label(self.current_id)
                self.itemconfigure(self.current_id, outline='green')
            self.reset_var()
    
    def mouse_move(self, event):
        mx, my = self.canvasx(event.x), self.canvasy(event.y)
        self.master.mouse_position_frame.set_position(mx * self.zoom_factor, my * self.zoom_factor)
    
    def save_cell(self, id, center, width, length, angle):
        self.cell_dict[id]["center"] = np.array(center) * self.zoom_factor
        self.cell_dict[id]["width"] = round(width * self.zoom_factor, 0)
        self.cell_dict[id]["length"] = round(length * self.zoom_factor, 0)
        self.cell_dict[id]["rotation"] = round(angle, 2)

    def reset_var(self):
        self.temp_item = None
        self.angle = None
        self.width = None
        self.length = None
        self.center = None
        # p0 is the cursor location when LB is pressed
        # p1 is the cursor location when LB is stopped/released
        # p2 is the cursor location when RB is pressed
        # p3 is calculated according to previous three points
        self.p0 = None
        self.p1 = None
        self.p2 = None
        self.p3 = None
        self.current_id = None
    def clear_cell_label(self):
        self.master.bouding_box_info_frame.clear_bounding_box_info()

    def update_cell_label(self, current_id):
        if current_id in self.cell_dict.keys():
            self.master.bouding_box_info_frame.set_bounding_box_info(self.cell_dict[current_id])
            
class MousePositionFrame(Frame):
    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)
        self.title = Label(self, text='Raw Image Coordinate')
        self.title.pack()
        self.coor = Label(self, text='(, )')
        self.coor.pack()
    def set_position(self, x, y):
        self.coor.config(text=f'({x:.2f}, {y:.2f})')

class BoundingBoxInfoFrame(Frame):
    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)
        self.cell_label = Label(self, text="Current bounding box: ", relief='flat')
        self.cell_label.grid(row=4, column=0, sticky=W, padx=10, pady=5)
        self.center_label = Label(self, text="center: ", relief='flat')
        self.center_label.grid(row=5, column=0, sticky=W, padx=10, pady=5)
        self.width_label = Label(self, text="width: ", relief='flat')
        self.width_label.grid(row=6, column=0, sticky=W, padx=10, pady=5)
        self.length_label = Label(self, text="length: ", relief='flat')
        self.length_label.grid(row=7, column=0, sticky=W, padx=10, pady=5)
        self.rotation_label = Label(self, text="rotation: ", relief='flat')
        self.rotation_label.grid(row=8, column=0, sticky=W, padx=10, pady=5)

    def set_bounding_box_info(self, info):
        self.center_label.config(text=f'center: ({info["center"][0]:.2f}, {info["center"][1]:.2f})')
        self.width_label.config(text=f'width: ({info["width"]:.2f})')
        self.length_label.config(text=f'length: ({info["length"]:.2f})')
        self.rotation_label.config(text=f'roation: ({info["rotation"]:.2f})')
    
    def clear_bounding_box_info(self):
        self.center_label.config(text='center: ')
        self.width_label.config(text='width: ')
        self.length_label.config(text='length: ')
        self.rotation_label.config(text='roation: ')
