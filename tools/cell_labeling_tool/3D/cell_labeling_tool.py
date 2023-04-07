from tkinter import *
from time import *
import tkinter.filedialog
from PIL import Image,ImageTk,ImageGrab
import numpy as np
import tkinter.messagebox
from collections import defaultdict
import csv
import platform
import os
import math
import uuid
from typing import Dict

from tkinter import ttk

from dataclasses import dataclass
import copy

mouseX = 0
mouseY = 0

@dataclass
class CellLabel:
    x: float
    y: float
    z: int
    r: float
    tag: StringVar

class CellLabelList:
    def __init__(self, cell_labels: Dict[str, CellLabel] = None):

        self.cell_labels = cell_labels if cell_labels is not None else {}
    
    def get_labels(self) -> Dict[str, CellLabel]:
        return self.cell_labels

    def get_labeled_frame(self, frame_num: int) -> CellLabel:
        return self.cell_labels.get(frame_num)
    
    def add_label(self, frame_num: int, cell_label: CellLabel) -> None:
        self.cell_labels[frame_num] = cell_label

    def remove_label(self, frame_num: int):
        del self.cell_labels[frame_num]
    
    def clear_labels(self):
        self.cell_labels.clear()

    def empty(self):
        return False if self.cell_labels else True

    def __repr__(self):
        return f"{self.cell_labels}"

class CellLabeling(Frame):
    def __init__(self,root):
        super().__init__(root)
        self.root = root
        self.w = self.root.winfo_width()
        self.h = self.root.winfo_height()

        self.x = self.y = 0
        self.x1 = self.y1 = None
        self.init_menu()
        self.init_component()
        self.new_flag = False
        self.delete_flag = False
        self.cell_dict = defaultdict(dict)
        self.frame_drawings = defaultdict(CellLabelList)
        self.plt = platform.system()

        self.z_scaling = 1

        self.right_clicked = False
        self.comboboxes = []

        self.show_labels = True

        self.filename = None
        self.load_file = None

        # bind mousewheel
        self.root.bind("<MouseWheel>", self._mouse_wheel)

        self.cell_centers = defaultdict(CellLabel)
    
    def init_menu(self):
        self.menu = Menu(self.root)
        self.root['menu'] = self.menu
        
        file_menu = Menu(self.menu,tearoff=0)
        self.menu.add_cascade(label='file',menu=file_menu)
        file_menu.add_command(label='open file',command=self.choose_file)
        file_menu.add_command(label='save cell data (csv file)',command=self.save_csv_file)
        file_menu.add_command(label="load file", command=self.reload_file)
        
    def init_component(self):
        self.new_button = Button(root, text="New bounding box",width=25,command=lambda i=1: self.choose_operation(i))
        self.new_button.grid(row=0, column=0, sticky=W, padx=10, pady=5)
        self.delete_button = Button(root, text="Delete bounding box",width=25,command=lambda i=2: self.choose_operation(i))
        self.delete_button.grid(row=1, column=0, sticky=W, padx=10, pady=5)

        # add new button to move to next frame
        self.next_button = Button(root, text="Move up Z-level",width=25,command=lambda:self.next_img())
        self.next_button.grid(row=2, column=0, sticky=W, padx=10, pady=5)
        self.previous_button = Button(root, text="Move down Z-level",width=25,command=lambda: self.previous_img())
        self.previous_button.grid(row=3, column=0, sticky=W, padx=10, pady=5)

        # add button to toggle the labels on an off
        self.toggle_labels = Button(root, text="Toggle label",width=25,command=lambda: self.toggle_label())
        self.toggle_labels.grid(row=4, column=0, sticky=W, padx=10, pady=5)
        
        self.mouse_label1 = Label(self.root,text="Current mouse location: ",relief='flat')
        self.mouse_label1.grid(row=5, column=0, sticky=W, padx=10, pady=5)
        self.mouse_label = Label(self.root,text=" ",relief='flat')
        self.mouse_label.grid(row=6, column=0, sticky=W, padx=10, pady=5)
        self.z_level = Label(self.root,text="current z-level: ",relief='flat')
        self.z_level.grid(row=7, column=0, sticky=W, padx=10, pady=5)
        self.dist_from_center_label = Label(self.root,text="center z-level: ",relief='flat')
        self.dist_from_center_label.grid(row=8, column=0, sticky=W, padx=10, pady=5)

        self.z_scaling_label = Label(self.root,text=f"z-scaling: 1",relief='flat')
        self.z_scaling_label.grid(row=9, column=0, sticky=W, padx=10, pady=5)
        self.z_scaling_input = Text(self.root, height=1 ,width=10)
        self.z_scaling_input.grid(row=10, column=0, sticky=W, padx=10, pady=5)
        self.z_scaling_submit = Button(self.root, height=1 ,width=15, text = "Change z-scaling", command=lambda: self.change_z_scaling())
        self.z_scaling_submit.grid(row=11, column=0, sticky=W, padx=10, pady=5)

        
    def choose_operation(self,i):
        if i ==1:
            self.new_flag = True
            self.delete_flag = False
            self.new_button.config(state=DISABLED)
            self.delete_button.config(state=NORMAL)
        if i==2:
            self.new_flag = False
            self.delete_flag = True
            self.delete_button.config(state=DISABLED)
            self.new_button.config(state=NORMAL)

    def toggle_label(self):
        self.show_labels = not self.show_labels

        if self.show_labels:
            self.update_frames()
        else:
            for combo_box in self.comboboxes:
                combo_box.destroy()


    def TextBoxUpdate(self, event, kwargs):
        cell_label = kwargs["cell_label"]
        cell_tag = kwargs["label_tag"]

        # update the cell labels by making a copy to the new id
        cell_label.tag = event.widget.get()

        cell_label_cp = copy.deepcopy(cell_label)

        self.frame_drawings[event.widget.get()].add_label(self.cur_frame, cell_label_cp)

        self.frame_drawings[cell_tag].remove_label(self.cur_frame)

        if self.frame_drawings[cell_tag].empty():
            del self.frame_drawings[cell_tag]

    def change_z_scaling(self):
        new_z_scale = self.z_scaling_input.get("1.0", "end-1c")
        try:
            self.z_scaling = float(new_z_scale)
            self.z_scaling_label.config(text = f"z-scaling: {new_z_scale}")
        except Exception as e:
            print(f"unable to set new z-sclaing {e}")
            self.z_scaling = 1.0

        if self.frame_drawings:
            # clear out all the cell labels
            for cell_id, cell_labels in self.frame_drawings.items():
                cell_labels.clear_labels()
            
            # readd cell labels with new scaling for center
            for cell_id, cell_center in self.cell_centers.items():
                self.frame_drawings[cell_id].add_label(cell_center.z, cell_center)
                self._create_sphere(cell_center.z, cell_center.tag, cell_center.z)
            
            # redraw cell labels
            self.update_frames()
        

    def reload_file(self):
        self.load_file = tkinter.filedialog.askopenfilename(title='choose file')
        img = ''
        if self.load_file is not None:
            with open(self.load_file) as csvfile:
                labels = csv.DictReader(csvfile, delimiter=',')
                for label in labels:
                    if label['file'] != img:
                        img = label['file']
                        self.img_file = img
                        self.init_img(img)
                    
                    tag = label["name"]
                    x = float(label['x']) * self.f
                    y = float(label['y']) * self.f
                    z = int(label['z'])
                    r = float(label['r'])
                    self.z_scaling = float(label['z_scaling'])
                    # update z_scaling_label
                    self.z_scaling_label.config(text = f"z-scaling: {self.z_scaling}")

                    # setup Cell labeling dictionary
                    self.frame_drawings[tag].add_label(z, CellLabel(x=x, y=y, z=z, r=r, tag=tag))

            self.init_canvas()
            self.update_frames()
        else:
            pass

    def draw_frame(self, canvas, z_level):
        cell_labels = list(self.frame_drawings.values())

        for cell_label in cell_labels:
            circle = cell_label.get_labeled_frame(z_level)

            if circle == None:
                continue

            x = circle.x
            y = circle.y
            r = circle.r
            tag = circle.tag
            new_frame_id = canvas.create_oval(x - r, y - r, x + r, y + r, outline="green", width=4, tags=["temp", tag])

            # generate tags only for center frame
            if z_level == self.cur_frame:

                # get only the id's for labels that exist on current frame
                ids = list(self.frame_drawings)

                self.comboboxes.append(ttk.Combobox(self.canvas, width = 5, values=ids))
                kwargs = {"label_tag": tag, "cell_label": circle}
                self.comboboxes[-1].bind("<<ComboboxSelected>>", lambda event, kwargs=kwargs: self.TextBoxUpdate(event, kwargs))
                self.comboboxes[-1].set(tag)
                self.comboboxes[-1].place(x=x,y=y)

    def _mouse_wheel(self, event):
        if event.num == 5 or event.delta == -120:
            self.next_img()
        elif event.num == 4 or event.delta == 120:
            self.previous_img()
        

    def update_frames(self):
        # clear temporary objects and redraw them
        self.canvas_prev.delete("temp")
        self.canvas.delete("temp")
        self.canvas_next.delete("temp")

        # remove the combo boxes so they can be updated
        for combo_box in self.comboboxes:
            combo_box.destroy()

        # update previous frame
        self.cur_img.seek(self.prev_frame)
        self.tk_img_1 = ImageTk.PhotoImage(self.resize_img(self.cur_img))
        self.canvas_prev.itemconfig(self.image_container_prev,image=self.tk_img_1)

        self.root.update()

        # update current frame
        self.cur_img.seek(self.cur_frame)
        self.tk_img_2 = ImageTk.PhotoImage(self.resize_img(self.cur_img))
        self.canvas.itemconfig(self.image_container,image=self.tk_img_2)

        self.root.update()

        # update next frame
        self.cur_img.seek(self.next_frame)
        self.tk_img_3 = ImageTk.PhotoImage(self.resize_img(self.cur_img))
        self.canvas_next.itemconfig(self.image_container_next,image=self.tk_img_3)
        
        # draw objects on frames
        self.draw_frame(self.canvas_prev, self.prev_frame)
        self.draw_frame(self.canvas, self.cur_frame)
        self.draw_frame(self.canvas_next, self.next_frame)

        self.root.update()

        self.cur_img.seek(self.cur_frame)

    def next_img(self):
        self.prev_frame = (self.cur_frame)
        self.cur_frame = (self.cur_frame + 1) % (self.cur_img.n_frames)
        self.next_frame = (self.cur_frame + 1) % (self.cur_img.n_frames)
        self.z_level.config(text=f"current z-level: {self.cur_frame}")

        self.update_frames()
    
    def previous_img(self):
        self.next_frame = self.cur_frame
        self.cur_frame =  (self.cur_frame - 1) if self.cur_frame > 0 else self.cur_img.n_frames - 1
        self.prev_frame = (self.cur_frame - 1) if (self.cur_frame - 1) >= 0 else self.cur_img.n_frames - 1

        self.z_level.config(text=f"current z-level: {self.cur_frame}")

        self.update_frames()

    def save_csv_file(self):
        self.csv_file = tkinter.filedialog.asksaveasfilename(
            defaultextension='.csv',               
            initialdir='',                         #default dir
            initialfile='cells.0',                 #init file name            
            title="Save cell data as csv file"                      
        )
        
        if self.csv_file is not None:
            f = open(self.csv_file,'w',encoding='utf-8',newline='')
            csv_writer = csv.writer(f)
            csv_writer.writerow(["file","name","x","y","z","r","z_scaling","split_alpha","opacity"])
            
            if self.frame_drawings:
                for cell_tag, cell_label_list in self.frame_drawings.items():
                    for z_level, cell_label in cell_label_list.get_labels().items():
                        name = cell_tag
                        x = cell_label.x / self.f
                        y = cell_label.y / self.f
                        z = z_level
                        r = cell_label.r
                        z_scaling = self.z_scaling
                        if self.filename is None:
                            self.filename = self.img_file
                        csv_writer.writerow([self.filename,name,x,y,z,r,z_scaling,"None", "None"])
            else:
                pass
            
            f.close()
        else:
            pass
    
    def exit(self):
        self.root.destroy()
    
    def reset_var(self):
        self.temp_item = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None
        self.x3 = None
        self.y3 = None
        self.angle = None
        self.width = None
        self.length = None
        self.center = None
        self.p0 = None
        self.p1 = None
        self.p2 = None
        self.p3 = None
        self.current_id = None
        self.cur_obj_tag = None
        self.right_clicked = False
        
    def init_mouse_binding(self):
        self.canvas.bind('<Motion>',self.mouse_move)
        self.canvas.bind('<ButtonPress-1>',self.left_press_down)
        
    def transfer_coordinate(self,x,y):
        #Translate mouse x,y screen coordinate to canvas coordinate
        mx = self.canvas.canvasx(x) 
        my = self.canvas.canvasy(y) 
        
        return mx,my
    
    def delete_warning(self):
        a=tkinter.messagebox.askokcancel('Warning', 'Delete this bounding box?')
        return a
    
    def draw_outside_error(self):
        tkinter.messagebox.showerror('Error','The center of the bounding box is outside of the image! Please draw again.')

    def mouse_move(self,event):
        mx,my = self.transfer_coordinate(event.x,event.y)
        self.mouse_label.config(text="(%s,%s)"%(mx,my),fg='blue')
    
    def left_motion(self,event):
        if self.new_flag is not True or self.right_clicked:
            return 
            
        if self.temp_item is not None:
            self.canvas.delete(self.temp_item)

        self.x1 = self.canvas.canvasx(event.x)
        self.y1 = self.canvas.canvasy(event.y) 

        r = math.dist([self.x0,self.y0], [self.x1,self.y1]) / 2
        x = (self.x0 + self.x1) / 2
        y = (self.y0 + self.y1) / 2
        self.temp_item = self.canvas.create_oval(x - r, y - r, x + r, y + r, outline="green", width=4, tag="temp")

        self.canvas.bind('<ButtonRelease-1>',self.stop_left_move)

        # right click to increase z-level for drawing sphere
        self.canvas.bind('<ButtonPress-3>', self.start_right_move)
    
    def left_press_down(self,event):
        self.reset_var()
            
        if self.new_flag is True:
            self.x0 = self.canvas.canvasx(event.x)
            self.y0 = self.canvas.canvasy(event.y)
            
            self.canvas.bind('<B1-Motion>',self.left_motion)
        
        elif self.delete_flag is True:
            mx = self.canvas.canvasx(event.x) 
            my = self.canvas.canvasy(event.y) 
            # get canvas object ID of where mouse pointer is 
            canvasobject = self.canvas.find_closest(mx, my, halo=5)

            if canvasobject != () and self.image_container != canvasobject[0]:
                self.canvas.itemconfig(canvasobject[0], outline='red')
                delete = self.delete_warning()
                if delete is True:
                    obj_tag = self.canvas.gettags(canvasobject[0])[1]

                    self.frame_drawings[obj_tag].remove_label(self.cur_frame)

                    if self.frame_drawings[obj_tag].empty():
                        del self.frame_drawings[obj_tag]

                    self.update_frames()

                else:
                    self.canvas.itemconfig(canvasobject[0], outline='green')
                    pass

    def stop_left_move(self,event):
        if self.new_flag is not True or self.right_clicked:
            return 

        self.x1 = self.canvas.canvasx(event.x) 
        self.y1 = self.canvas.canvasy(event.y) 
        
        if self.x1 > self.im_w:
            self.x1 = self.im_w
        elif self.x1 < 0:
            self.x1 = 0
        elif self.y1 > self.im_h:
            self.y1 = self.im_h
        elif self.y1 < 0:
            self.y1 = 0

        # draw using diameter
        r = math.dist([self.x0,self.y0], [self.x1,self.y1]) / 2
        x = (self.x0 + self.x1) / 2
        y = (self.y0 + self.y1) / 2

        self.cur_obj_tag = uuid.uuid4().hex
        self.temp_item = self.canvas.create_oval(x - r, y - r, x + r, y + r, outline="green", width=4, tags=["temp", self.cur_obj_tag])
        self.frame_drawings[self.cur_obj_tag].add_label(self.cur_frame, CellLabel(x=x, y=y, r=r, z=self.cur_frame, tag=self.cur_obj_tag))

        self.update_frames()

    def start_right_move(self,event):
        if self.new_flag is not True:
            return

        self.stop_left_move(event)

        self.right_clicked = True

        if self.plt == "Darwin":
            self.canvas.bind('<B2-Motion>', self.right_motion)
        else:
            self.canvas.bind('<B3-Motion>',self.right_motion)
    
    def _create_sphere(self, center, center_tag, start_frame):
        cur_cell = self.frame_drawings[center_tag].get_labeled_frame(start_frame)
            
        tag = center_tag
        center_x = cur_cell.x
        center_y = cur_cell.y

        if abs(start_frame - center) < 1:
            center_r = cur_cell.r
            slices_from_center = 1
            r = center_r
        
        else:
            center_r = math.sqrt((cur_cell.r ** 2) + (self.z_scaling * ((start_frame - center) ** 2)))
            # draw center
            self.frame_drawings[center_tag].add_label(center, CellLabel(x=center_x, y=center_y, r=center_r, z=center, tag=tag))
            r = center_r

            # delete start_frame since may not be included in final sphere (if it is then it will be set in while loop below)
            self.frame_drawings[center_tag].remove_label(start_frame)

            start_frame = center

        self.cell_centers[center_tag] = CellLabel(x=center_x, y=center_y, r=center_r, z=center, tag=tag)
        
        slices_from_center = 1

        while r > 0:
            try:
                radicand = (center_r ** 2) - ((self.z_scaling * ((center + slices_from_center) - center)) ** 2)
                if radicand < 0:
                    break
                else:
                    r = math.sqrt(radicand)

            except Exception as e:
                print(f"Exception: {e}")

            below_level = start_frame - slices_from_center
            above_level = start_frame + slices_from_center

            if  0 <= below_level < self.cur_img.n_frames:
                self.frame_drawings[tag].add_label(below_level, CellLabel(x=center_x, y=center_y, z=below_level, r=r, tag=tag))

            if 0 <= above_level < self.cur_img.n_frames:
                self.frame_drawings[tag].add_label(above_level, CellLabel(x=center_x, y=center_y, z=above_level, r=r, tag=tag))
            
            slices_from_center += 1

    def stop_right_move(self,event):
        if self.new_flag is not True:
            return 
    
        self.x2 = self.canvas.canvasx(event.x) 
        self.y2 = self.canvas.canvasy(event.y)

        self._create_sphere(self.z_new_center, self.cur_obj_tag, self.cur_frame)
        
        # for now update and draw circles on top and bottom frames
        self.update_frames()
    
    def right_motion(self,event):
        if self.new_flag is not True:
            return 
            
        self.x2 = self.canvas.canvasx(event.x)
        self.y2 = self.canvas.canvasy(event.y)

        # assuming z = 0 to be top of frame stack, then calculate the new relocated center
        cur_z_level = math.floor((mouseY - self.y1) / 10)
        z_diff = self.cur_frame + cur_z_level

        # z_diff < 0 means going up z-levels
        if z_diff < 0:
            self.z_new_center = max(0, z_diff)
        else:
            self.z_new_center = min(32, z_diff)


        # update distance away from current z-level
        self.dist_from_center_label.config(text=f"center z-level: {self.z_new_center}")

        if self.plt == "Darwin":
            self.canvas.bind('<ButtonRelease-2>',self.stop_right_move)
        else:
            self.canvas.bind('<ButtonRelease-3>',self.stop_right_move)
        
    
    def resize_img(self, img):
        return img.resize((self.rescaled_w, self.rescaled_h), Image.Resampling.NEAREST)

    def init_img(self, img):
        # initial frames to display
        self.prev_frame = 0
        self.cur_frame = 1
        self.next_frame = 2

        self.z_level.config(text=f"current z-level: {self.cur_frame}")
        self.cur_img = Image.open(img)
        self.im_w, self.im_h = self.cur_img.size
        self.NUM_IMGS = 3   # change depending on how many images to display

        # rescale images to fit into current frame
        self.rescaled_h = int(self.root.winfo_screenheight() / self.NUM_IMGS)
        self.f = (self.rescaled_h / self.im_h)
        # rescale based on ratio f to maintain aspect ratio given height
        self.rescaled_w = int(self.im_w * self.f)

    def init_canvas(self):
        # display frame 0
        self.cur_img.seek(self.prev_frame)
        self.tk_img_1 = ImageTk.PhotoImage(self.resize_img(self.cur_img))
        self.canvas_prev = Canvas(self.root,width=self.rescaled_w,height=self.rescaled_h)
        self.canvas_prev.place(x=self.w // 2, y=0)
        self.image_container_prev = self.canvas_prev.create_image(0,0,anchor="nw",image=self.tk_img_1)

        self.root.update()

        # display frame 1
        self.cur_img.seek(self.cur_frame)
        self.tk_img_2 = ImageTk.PhotoImage(self.resize_img(self.cur_img))
        self.canvas = Canvas(self.root,width=self.rescaled_w,height=self.rescaled_h)
        self.canvas.place(x=self.canvas_prev.winfo_x(), y=self.canvas_prev.winfo_y() + self.rescaled_h + 10)
        self.image_container = self.canvas.create_image(0,0,anchor="nw",image=self.tk_img_2)

        self.root.update()
        
        # display frame 2
        self.cur_img.seek(self.next_frame)
        self.tk_img_3 = ImageTk.PhotoImage(self.resize_img(self.cur_img))
        self.canvas_next = Canvas(self.root,width=self.rescaled_w,height=self.rescaled_h)
        self.canvas_next.place(x=self.canvas.winfo_x(), y=self.canvas.winfo_y() + self.rescaled_h + 10)
        self.image_container_next = self.canvas_next.create_image(0,0,anchor="nw",image=self.tk_img_3)
        
        self.root.update()
        
        self.cur_img.seek(self.cur_frame)

        self.reset_var()
        self.init_mouse_binding()

    def choose_file(self):
        self.filename = tkinter.filedialog.askopenfilename(title='choose file')
        if self.filename is not None:
            self.init_img(self.filename)
            self.init_canvas()
        else:
            pass

    
def close_escape(event=None):
    root.destroy()

def get_mouse_pos(event):
    global mouseX 
    global mouseY

    mouseY = event.y
    mouseX = event.x

if __name__=='__main__':
    root = Tk()
    root.attributes('-fullscreen', True)
    root.title('Cell Labeling Tool')
    root.update_idletasks()
    screen_w = root.winfo_width()
    screen_h = root.winfo_height()
    root.geometry(f"{screen_w}x{screen_h}")
    root.bind("<Escape>", close_escape)
    root.bind("<Motion>", get_mouse_pos)
    cell_labeling = CellLabeling(root)
    root.mainloop()