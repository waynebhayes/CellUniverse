from tkinter import *
from time import *
import tkinter.filedialog
from PIL import Image, ImageTk, ImageGrab
import numpy as np
import tkinter.messagebox
from collections import defaultdict
import csv
import platform
import os
from point import Point
import math

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 640


class CellLabeling(Frame):
    ####################################################
    #              REGISTER ALL COMPONENTS
    ####################################################
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.w = WINDOW_WIDTH
        self.h = WINDOW_HEIGHT
        self.init_menu()
        self.init_component()
        self.new_flag = False
        self.delete_flag = False
        self.cell_dict = defaultdict(dict)
        self.plt = platform.system()

    def init_menu(self):
        self.menu = Menu(self.root)
        self.root['menu'] = self.menu

        file_menu = Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label='file', menu=file_menu)
        file_menu.add_command(label='open file', command=self.choose_file)
        file_menu.add_command(
            label='save cell data (csv file)', command=self.save_csv_file)
        file_menu.add_command(label='load cell data (csv file)', command=self.load_csv_file)
        file_menu.add_command(
            label='save current cell image', command=self.get_screenshot)
        
        # file_menu.add_command(label='exit',command=self.exit)

    def init_component(self):
        self.new_button = Button(root, text="New bounding box",
                                 width=25, command=lambda i=1: self.choose_operation(i))
        self.new_button.grid(row=0, column=0, sticky=W, padx=10, pady=5)
        self.delete_button = Button(
            root, text="Delete bounding box", width=25, command=lambda i=2: self.choose_operation(i))
        self.delete_button.grid(row=1, column=0, sticky=W, padx=10, pady=5)

        self.mouse_label1 = Label(
            self.root, text="Current mouse location: ", relief='flat')
        self.mouse_label1.grid(row=2, column=0, sticky=W, padx=10, pady=5)
        self.mouse_label = Label(self.root, text=" ", relief='flat')
        self.mouse_label.grid(row=3, column=0, sticky=W, padx=10, pady=5)
        self.cell_label = Label(
            self.root, text="Current bounding box: ", relief='flat')
        self.cell_label.grid(row=4, column=0, sticky=W, padx=10, pady=5)
        self.center_label = Label(self.root, text="center: ", relief='flat')
        self.center_label.grid(row=5, column=0, sticky=W, padx=10, pady=5)
        self.width_label = Label(self.root, text="width: ", relief='flat')
        self.width_label.grid(row=6, column=0, sticky=W, padx=10, pady=5)
        self.length_label = Label(self.root, text="length: ", relief='flat')
        self.length_label.grid(row=7, column=0, sticky=W, padx=10, pady=5)
        self.rotation_label = Label(
            self.root, text="rotation: ", relief='flat')
        self.rotation_label.grid(row=8, column=0, sticky=W, padx=10, pady=5)

    ####################################################
    #              MENU - Open File
    ####################################################
    def resize_img(self, img):
        self.init_w, self.init_h = img.size
        self.f = min([self.w/self.init_w, self.h/self.init_h])
        return img.resize((int(self.init_w*self.f), int(self.init_h*self.f)), Image.Resampling.LANCZOS)

    def init_mouse_binding(self):
        self.canvas.bind('<Motion>', self.mouse_move)
        self.canvas.bind('<ButtonPress-1>', self.left_press_down)

    def choose_file(self):
        self.filename = tkinter.filedialog.askopenfilename(title='choose file')
        if self.filename is not None:
            img_open = Image.open(self.filename)
            resized = self.resize_img(img_open)
            self.resize_w, self.resize_h = resized.size
            self.tk_img = ImageTk.PhotoImage(resized)
            self.canvas = Canvas(
                self.root, width=self.resize_w, height=self.resize_h)
            self.canvas.place(x=300, y=0)
            self.image = self.canvas.create_image(
                0, 0, anchor="nw", image=self.tk_img)
            self.reset_var()
            self.cell_dict.clear()
            self.init_mouse_binding()
        else:
            pass

    ####################################################
    #              MENU - Save Cell Data
    ####################################################
    def dec_to_bin(self, n, digit=2):
        result = 'b'
        for i in range(digit-1, -1, -1):
            result += str((n >> i) & 1)
        return result

    def save_csv_file(self):
        self.csv_file = tkinter.filedialog.asksaveasfilename(
            defaultextension='.csv',
            # filetypes=[('txt Files', '*.txt'), #other file type
            #   ('pkl Files', '*.pkl'),
            #   ('All Files', '*.*')],
            initialdir='',  # default dir
            initialfile='cells.0',  # init file name
            # parent=self.master,
            title="Save cell data as csv file"
        )
        # print(self.csv_file)

        if self.csv_file is not None:
            f = open(self.csv_file, 'w', encoding='utf-8', newline='')
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                ["file", "name", "x", "y", "width", "length", "rotation", "split_alpha", "opacity"])

            if self.cell_dict != {}:
                cell_num = len(self.cell_dict.keys())
                digit_n = len(bin(cell_num)[2:])
                data = list(self.cell_dict.values())
                filename = os.path.basename(self.filename)
                for i in range(cell_num):
                    name = self.dec_to_bin(i, digit=digit_n)
                    x = data[i]["center"][0]
                    y = data[i]["center"][1]
                    width = data[i]["width"]
                    length = data[i]["length"]
                    rotation = data[i]["rotation"]
                    # print(name)
                    csv_writer.writerow(
                        [filename, name, x, y, width, length, rotation, "None", "None"])
            else:
                pass

            f.close()
        else:
            pass
    ####################################################
    #              MENU - Load Cell Data
    ####################################################
    def dec_to_bin(self, n, digit=2):
        result = 'b'
        for i in range(digit-1, -1, -1):
            result += str((n >> i) & 1)
        return result

    def load_csv_file(self):
        def rectangle_coordinates(c, w, h, angle):

            # calculate the half width and half height
            hw = w/2
            hh = h/2

            # u is the unit vector along the length direction
            u = np.array([hh * math.cos(angle), hh * math.sin(angle)])
            # b is the unit vector along the width direction
            v = np.array([-hw * math.sin(angle), hw * math.cos(angle)])

            # calculate the coordinates of the four corners
            p1 = c - u + v
            p2 = c + u + v
            p3 = c + u - v
            p4 = c - u - v

            # return the coordinates as a list of tuples
            return [p1, p2, p3, p4]
                
        def flush_canvas_objects(self):
            """
            Remove all objects from the given tkinter canvas,
            except the image loaded.
            """
            for item in self.canvas.find_all():
                # check if the item is the loaded image
                if item != self.image:
                    self.canvas.delete(item)

        self.csv_file = tkinter.filedialog.askopenfilename(
            defaultextension='.csv',
            # filetypes=[('txt Files', '*.txt'), #other file type
            #   ('pkl Files', '*.pkl'),
            #   ('All Files', '*.*')],
            initialdir='',  # default dir
            initialfile='cells.0',  # init file name
            # parent=self.master,
            title="Save cell data as csv file"
        )

        if self.csv_file is not None:
            f = open(self.csv_file, 'r', encoding='utf-8', newline='')
            self.cell_dict.clear()
            flush_canvas_objects(self)
            csv_reader = csv.DictReader(f)

            for row in csv_reader:
                x, y, w, l, angle = float(row['x']), float(row['y']), float(row['width']), float(row['length']), float(row['rotation'])
                x, y, w, l = x * self.f, y * self.f, w * self.f, l * self.f
                coords = rectangle_coordinates(np.array([x, y]), w, l, angle)
                object_id  = self.canvas.create_polygon(*coords[0], *coords[1], *coords[2], *coords[3], fill='', outline='green', width=3)
                self.save_cell(object_id, (x, y), w, l, angle)

            f.close()

        
    ####################################################
    #      MENU - Save Current Cell Image
    ####################################################
    # Need to resacle your monitor's display setting to 100%
    def get_screenshot(self):
        self.canvas.update()
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.resize_w
        y1 = y + self.resize_h
        # print(x, y, x1, y1)
        self.save_image = tkinter.filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[("PNG", ".png"),  # other file type
                       ('JPG', '.jpg')],
            initialdir='',  # default dir
            initialfile='cells.0.labeled',  # init file name
            title="Save current cell image"
        )
        ImageGrab.grab().crop((x, y, x1, y1)).save(self.save_image)

    ####################################################
    #     Callback function New/Delete bounding box
    ####################################################

    def choose_operation(self, i):
        if i == 1:
            self.new_flag = True
            self.delete_flag = False
            self.new_button.config(state=DISABLED)
            self.delete_button.config(state=NORMAL)
        if i == 2:
            self.new_flag = False
            self.delete_flag = True
            self.delete_button.config(state=DISABLED)
            self.new_button.config(state=NORMAL)

    ####################################################
    #      Callback functions for canvas
    ####################################################
    def left_press_down(self, event):
        if self.temp_item is not None:
            self.canvas.delete(self.temp_item)

        if self.new_flag is True:
            self.p0 = Point([self.canvas.canvasx(event.x),
                            self.canvas.canvasy(event.y)])
            self.canvas.bind('<B1-Motion>', self.left_motion)

        elif self.delete_flag is True:
            mx = self.canvas.canvasx(event.x)
            my = self.canvas.canvasy(event.y)
            # get canvas object ID of where mouse pointer is
            canvasobject = self.canvas.find_closest(mx, my, halo=5)

            # print(canvasobject)
            # self.canvas.delete(canvasobject)

            if canvasobject != () and self.image != canvasobject[0]:
                self.update_cell_label(canvasobject[0])
                delete = self.delete_warning()
                if delete is True:
                    self.canvas.delete(canvasobject[0])
                    del self.cell_dict[canvasobject[0]]
                    #print("delete: {}".format(canvasobject[0]))
                    # print(self.cell_dict)
                else:
                    pass
    
    def left_motion(self, event):
        if self.new_flag is not True:
            return
        if self.temp_item is not None:
            self.canvas.delete(self.temp_item)
        self.p1 = Point([self.canvas.canvasx(event.x),
                        self.canvas.canvasy(event.y)])
        self.temp_item = self.canvas.create_line(
            *self.p0, *self.p1, fill='green', width=3)
        self.canvas.bind('<ButtonRelease-1>', self.stop_left_move)

    def stop_left_move(self, event):
        if self.new_flag is not True:
            return
        if self.temp_item is not None:
            self.canvas.delete(self.temp_item)

        self.p1 = Point([self.canvas.canvasx(event.x),
                        self.canvas.canvasy(event.y)])

        # print(self.p0)
        # print(self.p1)

        self.temp_item = self.canvas.create_line(
            *self.p0, *self.p1, fill='green', width=3)
        if self.plt == "Darwin":
            # 2 for MacOS Right-Click, 3 for Windows
            self.canvas.bind('<ButtonPress-2>', self.start_right_move)
        else:
            self.canvas.bind('<ButtonPress-3>', self.start_right_move)

    def start_right_move(self, event):
        if self.new_flag is not True:
            return
        self.p2 = Point([self.canvas.canvasx(event.x),
                        self.canvas.canvasy(event.y)])
        if self.plt == "Darwin":
            self.canvas.bind('<B2-Motion>', self.right_motion)
        else:
            self.canvas.bind('<B3-Motion>', self.right_motion)
    
    def right_motion(self, event):
        if self.new_flag is not True:
            return
        if self.temp_item is not None:
            self.canvas.delete(self.temp_item)

        self.p2 = Point([self.canvas.canvasx(event.x),
                        self.canvas.canvasy(event.y)])

        self.calculate_vertices2(self.p0, self.p1, self.p2)

        self.temp_item = self.canvas.create_polygon(self.p0[0], self.p0[1],
                                                    self.p1[0], self.p1[1],
                                                    self.p2[0], self.p2[1],
                                                    self.p3[0], self.p3[1],
                                                    fill='', outline='green', width=3)
        if self.plt == "Darwin":
            self.canvas.bind('<ButtonRelease-2>', self.stop_right_move)
        else:
            self.canvas.bind('<ButtonRelease-3>', self.stop_right_move)

    def stop_right_move(self, event):
        if self.new_flag is not True:
            return
        if self.temp_item is not None:
            self.canvas.delete(self.temp_item)

        self.p2 = Point([self.canvas.canvasx(event.x),
                        self.canvas.canvasy(event.y)])

        # print(self.p0)

        self.calculate_vertices2(self.p0, self.p1, self.p2)
        self.current_id = self.canvas.create_polygon(self.p0[0], self.p0[1],
                                                     self.p1[0], self.p1[1],
                                                     self.p2[0], self.p2[1],
                                                     self.p3[0], self.p3[1],
                                                     fill='', outline='green', width=3)
        # print(self.current_id)

        # check the center inside the image
        if self.center[0] > self.resize_w or self.center[0] < 0 or \
                self.center[1] > self.resize_h or self.center[1] < 0:
            self.draw_outside_error()
            self.canvas.delete(self.current_id)
        else:
            # print("add:{}".format(self.current_id))
            # save to cell_dict
            self.save_cell(self.current_id, self.center, self.width, self.length, self.angle)
            # print(self.cell_dict)

            # update cell label
            self.update_cell_label(self.current_id)

        self.reset_var()
    
    def calculate_vertices2(self, p0, p1, p2):
        self.angle = self.rotation_angle(p0, p1)
        self.p2 = self.map_p2(p0, p1, p2)
        p2 = self.p2
        self.width = self.compute_edge_length(p1, p2)
        self.length = self.compute_edge_length(p0, p1)
        self.center = ((p0.x + p2.x) / 2, (p0.y + p2.y) / 2)
        self.p3 = ((p0.x + p2.x - p1.x), (p0.y + p2.y - p1.y))

    def rotation_angle(self, p0, p1):
        # compute rotation angle
        x = p1.x - p0.x
        y = p1.y - p0.y
        # print(x)
        angle = np.degrees(np.arctan2(y, x))
        if angle < 0:
            angle = 360 + angle
        angle = np.radians(angle)
        return angle
    
    def compute_edge_length(self, p0, p1):
        x = abs(p1.x - p0.x)
        y = abs(p1.y - p0.y)
        length = np.sqrt(x**2 + y**2)
        return length
    
    def map_p2(self, p0, p1, p2):
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

        mapped_p2 = Point([p1.x + distance * perpendicular_vec[0], p1.y + distance * perpendicular_vec[1]])

        # print(f'Point 0: {p0}, Point 1:{p1}, Point 2:{p2}')
        # print(f'equation: {a}x+{b}y+{c}=0')
        # print(f'is_clockwise: {p2_is_clockwise}')
        # print(f'distance: {distance}')
        # print(f'perpendicular_vec: {perpendicular_vec}')
        return mapped_p2

    ####################################################
    #            Helper Functions
    ####################################################

    def exit(self):
        self.root.destroy()

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

    def transfer_coordinate(self, x, y):
        # Translate mouse x,y screen coordinate to canvas coordinate
        mx = self.canvas.canvasx(x)
        my = self.canvas.canvasy(y)
        # Translate canvas coordinate to original image coordinate
        mx = mx / self.f
        my = my / self.f

        return mx, my

    def delete_warning(self):
        a = tkinter.messagebox.askokcancel(
            'Warning', 'Delete this bounding box?')
        return a

    def mouse_move(self, event):
        mx, my = self.transfer_coordinate(event.x, event.y)
        self.mouse_label.config(text="(%s,%s)" % (mx, my), fg='blue')

    def draw_outside_error(self):
        tkinter.messagebox.showerror(
            'Error', 'The center of the bounding box is outside of the image! Please draw again.')

    def update_cell_label(self, current_id):
        if current_id in self.cell_dict.keys():
            self.center_label.config(text="center: (%.2f,%.2f)" % (self.cell_dict[current_id]["center"][0],
                                                                   self.cell_dict[current_id]["center"][1]))
            self.width_label.config(text="width: %s" %
                                    (self.cell_dict[current_id]["width"]))
            self.length_label.config(text="length: %s" % (
                self.cell_dict[current_id]["length"]))
            self.rotation_label.config(text="rotation: %s" % (
                self.cell_dict[current_id]["rotation"]))

    def save_cell(self, id, center, width, length, angle):
        self.cell_dict[id]["center"] = np.array(center)/self.f
        self.cell_dict[id]["width"] = round(width/self.f, 0)
        self.cell_dict[id]["length"] = round(length/self.f, 0)
        self.cell_dict[id]["rotation"] = round(angle, 2)
   


if __name__ == '__main__':
    root = Tk()
    root.title('Cell Labeling Tool')
    root.geometry(f'{WINDOW_WIDTH}x{WINDOW_HEIGHT}')
    cell_labeling = CellLabeling(root)
    root.mainloop()
