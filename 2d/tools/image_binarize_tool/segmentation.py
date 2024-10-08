# importing the tkinter module and PIL 
# that is pillow module 
import os
from tkinter import *
from PIL import ImageTk, Image 
from tkinter import messagebox, filedialog
import numpy as np
import colorsys
import PIL
from threading import Thread
import cv2
import skimage.color
import skimage.io
import skimage.viewer
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  NavigationToolbar2Tk) 
import pathlib

from multiprocessing import Process
import time
import threading

inited = 0
saturation = 0.1
DEFAULT_FONT = ('Helvetica', 14)

count = 0

begin_main = 0



def number_of_pages(begin_frame, end_frame, frame_intervals):
    pages = (end_frame - begin_frame) / (3*frame_interval)
    return pages

class ThreadedTask(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        global begin

        im_phase = skimage.io.imread("{}.png".format(0))
        hist_phase, bins_phase = skimage.exposure.histogram(im_phase)

        # Use matplotlib to make a pretty plot of histogram data

        fig, ax = plt.subplots(1, 1)
    
        ax.set_xlabel('pixel value')

        ax.set_ylabel('count')

        ratio = 0.001

        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        _ = ax.fill_between(bins_phase, hist_phase, alpha=0.75)
        plt.show()


def threshold_button_clicked():
    ThreadedTask().start()

class InputDialog:
    def __init__(self, input_dir):
        self.dialog_window = Toplevel()

        self.input_dir = input_dir
        self.ok_clicked = False
        self.pattern_input = StringVar()
        self.start_input = IntVar()
        self.end_input = IntVar()
        self.frame_interval = IntVar()
        self.images = []

        pattern_input_label = Label(master=self.dialog_window, text="Input filename pattern (e.g. \"image%%03d.png\")", font=DEFAULT_FONT)
        pattern_input_label.grid(row=0, column=0, padx=10, pady=10, sticky=W)
        pattern_input_entry = Entry(master=self.dialog_window, width=20, textvariable=self.pattern_input, font=DEFAULT_FONT)
        pattern_input_entry.grid(row=0, column=1, padx=10, pady=1, sticky=W+E)

        start_input_label = Label(master=self.dialog_window, text="Input start frame number", font=DEFAULT_FONT)
        start_input_label.grid(row=1, column=0, padx=10, pady=10, sticky=W)
        start_input_entry = Spinbox(master=self.dialog_window, from_=0, increment=1, width=20, textvariable=self.start_input, font=DEFAULT_FONT)
        start_input_entry.grid(row=1, column=1, padx=10, pady=1, sticky=W+E)

        end_input_label = Label(master=self.dialog_window, text="Input end frame number", font=DEFAULT_FONT)
        end_input_label.grid(row=2, column=0, padx=10, pady=10, sticky=W)
        end_input_entry = Spinbox(master=self.dialog_window, from_=0, increment=1, width=20, textvariable=self.end_input, font=DEFAULT_FONT)
        end_input_entry.grid(row=2, column=1, padx=10, pady=1, sticky=W+E)

        frame_interval_label = Label(master=self.dialog_window, text="Input frame interval", font=DEFAULT_FONT)
        frame_interval_label.grid(row=3, column=0, padx=10, pady=10, sticky=W)
        frame_interval_entry = Spinbox(master=self.dialog_window, from_=0, increment=1, width=20, textvariable=self.frame_interval, font=DEFAULT_FONT)
        frame_interval_entry.grid(row=3, column=1, padx=10, pady=1, sticky=W+E)

        button_frame = Frame(master=self.dialog_window)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        ok_button = Button(master=button_frame, text="OK", font=DEFAULT_FONT, command=self.on_ok_button)
        ok_button.grid(row=0, column=0)
        cancel_button = Button(master=button_frame, text="Cancel", font=DEFAULT_FONT, command=self.on_cancel_button)
        cancel_button.grid(row=0, column=1)

    def validate_input(self):
        self.images = []
        if self.start_input.get() > self.end_input.get():
            raise ValueError('Invalid interval: start frame must be less than or equal to end frame')
        for i in range(self.start_input.get(), self.end_input.get() + 1):
            file = self.input_dir / pathlib.Path(self.pattern_input.get() % i)
            if file.exists() and file.is_file():
                self.images.append(file)
            else:
                raise ValueError(f'Input file not found "{file}"')

    def get_input(self):
        return self.pattern_input.get(), self.start_input.get(), self.end_input.get(), self.images, self.frame_interval.get()

    def was_ok_clicked(self):
        return self.ok_clicked

    def show(self):
        self.dialog_window.grab_set()
        self.dialog_window.wait_window()

    def on_ok_button(self):
        try:
            self.validate_input()
            self.ok_clicked = True
            self.dialog_window.destroy()
        except ValueError as e:
            self.images = []
            messagebox.showerror(title="Error", message=e)

    def on_cancel_button(self):
        self.dialog_window.destroy()

def forward(): 
    # GLobal variable so that we can have 
    # access and change the variable 
    # whenever needed 
    global begin
    global begin_main
    global end
    global current_page
    global image1 
    global image2 
    global image3 
    global image4 
    global image5 
    global image6 
    global image7 
    global image8 
    global scale_widget
    global button_forward 
    global button_back 
    global button_exit 
    global button_set
    global images
    global frame_interval
    global end_frame
    global total_pg_text
    global page_slider

    current_page += 1
    if page_slider.get() < current_page: 
        page_slider.set(current_page)


    begin      += frame_interval * 3 
    begin_main += frame_interval * 3 
    end        += frame_interval * 3 
    update_binary_image(begin, end, scale_widget.get())
    setup_images(begin, end)
    

    # This is for clearing the screen so that 
    # our next image can pop up 
    image1.grid_forget() 
    image2.grid_forget() 
    image3.grid_forget() 
    image4.grid_forget() 
    image5.grid_forget() 
    image6.grid_forget() 
    image7.grid_forget() 
    image8.grid_forget() 
    button_forward.grid_forget()
    button_back.grid_forget()
    total_pg_text.grid_forget()
    draw_grid()

    update_frame_text_threshold()
    
    if end_frame - end < 3 * frame_interval + 1: 
        button_forward = Button(root, text="Forward", command=forward, state=DISABLED) 
    else: 
        button_forward = Button(root, text="Forward", command=forward) 
    # back button 
    button_back = Button(root, text="Back",command=back) 
    button_forward.grid(row=7, column=2,sticky = "w") 
    button_back.grid(row=7, column=1,sticky = "e") 
  
def back(): 
    # We will have global variable to access these 
    # variable and change whenever needed 
    global begin 
    global begin_main
    global end
    global image1 
    global current_page
    global image2 
    global image3 
    global image4 
    global image5 
    global image6 
    global image7 
    global image8 
    global scale_widget
    global button_forward 
    global button_back 
    global button_exit 
    global button_set
    global frame_interval
    global total_pg_text
    global page_slider

    begin      -= frame_interval * 3 
    begin_main -= frame_interval * 3 
    end        -= frame_interval * 3 

    current_page -= 1

    if page_slider.get() > current_page:
        page_slider.set(current_page)

    update_binary_image(begin, end, scale_widget.get())
    setup_images(begin, end)

    # This is for clearing the screen so that 
    # our next image can pop up 
    image1.grid_forget() 
    image2.grid_forget() 
    image3.grid_forget() 
    image4.grid_forget() 
    image5.grid_forget() 
    image6.grid_forget() 
    image7.grid_forget() 
    image8.grid_forget() 
    button_forward.grid_forget()
    button_back.grid_forget()
    total_pg_text.grid_forget()
    draw_grid()
    update_frame_text_threshold()
    # back button disabled 
    if begin <= frame_interval +1:
        button_back = Button(root, text="Back", state=DISABLED) 
    else:
        button_back = Button(root, text="Back", command=back) 
    button_forward = Button(root, text="Forward", command=forward) 
    button_forward.grid(row=7, column=2,sticky = "w") 
    button_back.grid(row=7, column=1,sticky = "e") 

def calculate_otzu_threshold():
    global images
    global begin_frame
    global end_frame
    count = 0
    total_threshold = 0
    
    for i in range(begin_frame, end_frame):
        src = cv2.imread(str(images[i].resolve()), cv2.IMREAD_GRAYSCALE)
        th, dst = cv2.threshold(src, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        total_threshold += th
        count += 1
    avg = total_threshold/count
    return avg

def calculate_otzu_threshold_interval(begin, end):
    global images
    global begin_frame
    global end_frame
    count = 0
    total_threshold = 0
    
    for i in range(begin, end):
        src = cv2.imread(str(images[i].resolve()), cv2.IMREAD_GRAYSCALE)
        th, dst = cv2.threshold(src, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        total_threshold += th
        count += 1
    avg = total_threshold/count
    return avg

def update_frame_text_threshold():

    global frame_interval 

    global current_page
    global page_threshold_dict
    global frame_interval
    global q1_text
    global q2_text
    global q3_text
    global q4_text


    if current_page > 1:
        # calculating linear threshold to display on the GUI 
        page_number = current_page
        previous_page_threshold = page_threshold_dict[page_number - 1][1]
        current_page_threshold  = page_threshold_dict[page_number][1]

        begin_frame = (current_page - 1) * (frame_interval * 3)
        end_frame = (begin_frame + frame_interval * 3)

        threshold_increase_per_frame = (current_page_threshold - previous_page_threshold)/(frame_interval * 3)

        top_left_frame_smooth_thresh = threshold_increase_per_frame * ((begin_frame) - page_threshold_dict[page_number - 1][3]) + previous_page_threshold

        top_right_frame_smooth_thresh = threshold_increase_per_frame * ((begin_frame + frame_interval) - page_threshold_dict[page_number - 1][3]) + previous_page_threshold

        bottom_left_frame_smooth_thresh = threshold_increase_per_frame * ((begin_frame + (frame_interval*2)) - page_threshold_dict[page_number - 1][3]) + previous_page_threshold
        
        bottom_right_frame_smooth_thresh = scale_widget.get()

        q1_text.grid_forget()
        q2_text.grid_forget() 
        q3_text.grid_forget() 
        q4_text.grid_forget()

        q1_text = Label(root, text="Frame {} ({:.2f})".format(begin_frame, top_left_frame_smooth_thresh))

        q2_text = Label(root, text="Frame {} ({:.2f})".format(begin_frame+frame_interval*1, top_right_frame_smooth_thresh))

        q3_text = Label(root, text="Frame {} ({:.2f})".format(begin_frame+frame_interval*2, bottom_left_frame_smooth_thresh))
        
        q4_text = Label(root, text="Frame {} ({:.2f})".format(begin_frame+frame_interval*3, bottom_right_frame_smooth_thresh))

        q1_text.grid(row=0, column=0, columnspan=2)
        q2_text.grid(row=0, column=2, columnspan=2) 
        q3_text.grid(row=2, column=0, columnspan=2)
        q4_text.grid(row=2, column=2, columnspan=2)



def update_binary_image(begin,end,threshold): 
    global second_image
    global processed_images
    global images
    global frame_interval 
    global current_page
    global page_threshold_dict
    global frame_interval
    global q1_text
    global q2_text
    global q3_text
    global q4_text


    processed_images = []
    for i in range(begin,end+1,frame_interval):
        # Read image
        src = cv2.imread(str(images[i].resolve()), cv2.IMREAD_GRAYSCALE)

        # Set threshold and maxValue
        thresh = threshold
        maxValue = 255 

        # Threshold
        th, dst = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY_INV);      
        
        im_pil = Image.fromarray(dst)
        second_image = ImageTk.PhotoImage(im_pil)
        processed_images.append(second_image)
    return second_image

def get_begin():
    global begin
    return begin

def update_page_threshold_dict(threshold, current_page):
    global page_threshold_dict
    global total_pages
    
    page_threshold_dict[current_page][1] = threshold

    for i in range(current_page+1, int(total_pages) + 1):
        if page_threshold_dict[i][0] == False:
            page_threshold_dict[i][1] = threshold
        else:
            break

def display_segmented_image(dum):
    global scale_widget
    global s
    global image2
    global image4
    global image6
    global image8
    global thresh_text
    global current_page
    global page_threshold_dict
    global frame_interval
    global q1_text
    global q2_text
    global q3_text
    global q4_text

    update_frame_text_threshold()
    
    image2.grid_forget()
    image4.grid_forget()
    image6.grid_forget()
    image8.grid_forget()
    thresh_text.grid_forget()

    s = scale_widget.get()
    update_binary_image(begin, end, s)

    update_page_threshold_dict(s, current_page)
    #assigning text
    display_threshold() 
    #assigning images
    image2 = Label(image=processed_images[0])
    image4 = Label(image=processed_images[1])
    image6 = Label(image=processed_images[2])
    image8 = Label(image=processed_images[3])
    
    #grid-ing
    image2.grid(row=1, column=1)#, sticky="e") 
    image4.grid(row=1, column=3) 
    image6.grid(row=3, column=1) 
    image8.grid(row=3, column=3) 
 
def setup_images(begin, end):
    global List_images
    global images
    global frame_interval
    List_images = [] 
    for i in range(begin, end+1, frame_interval): #this is the bug
        List_images.append(ImageTk.PhotoImage(Image.open(images[i])))
  
def plot_a_graph():
    global begin
    global images
    global input_dir
    global current_page
    global page_threshold_dict

    im_phase = skimage.io.imread("{}/{}.png".format(input_dir,begin))
    hist_phase, bins_phase = skimage.exposure.histogram(im_phase)
    # Use matplotlib to make a pretty plot of histogram data
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('pixel value')
    ax.set_ylabel('count')
    ratio = 0.001
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    _ = ax.fill_between(bins_phase, hist_phase, alpha=0.75)

    xmin, xmax, ymin, ymax = plt.axis()
    total_threshold = 0

    src = cv2.imread(str(images[begin].resolve()), cv2.IMREAD_GRAYSCALE)
  
    th = page_threshold_dict[current_page][1] 

    _ = ax.plot([th,th], [0, ymax], color="red", label = "Selected threshold")
    plt.legend(loc = "upper right")
    plt.show()

def display_threshold():
    global page_threshold_dict
    global total_pages
    global thresh_text
    thresh_string = ""
    
    for i in range(1, int(total_pages) + 1):
        thresh_string += "Page {} threshold = {}\n".format(i, page_threshold_dict[i][1]) 

    thresh_text = Label(root, text = thresh_string)
    thresh_text.grid(row=0, column=4, rowspan=7)

def draw_grid():
    global image1
    global image2
    global image3
    global image4
    global image5
    global image6
    global image7
    global image8
    global q1_text
    global q2_text
    global q3_text
    global q4_text
    global begin_frame
    global end_frame
    global begin
    global end
    global count
    global thres
    global total_pages
    global current_page
    global total_pg_text 
    global page_threshold_dict
    global thresh_text
    global scale_widget
    
    #setting visted bool to True
    page_threshold_dict[current_page][0] = True
    scale_widget.set(page_threshold_dict[current_page][1])

    if "image1" in globals():
        image1.grid_forget()
        image2.grid_forget()
        image3.grid_forget()
        image4.grid_forget()
        image5.grid_forget()
        image6.grid_forget()
        image7.grid_forget()
        image8.grid_forget()
    #print("forgeting q1_text from draw_grid()")
    if "q1_text" in globals():
        q1_text.grid_forget()
        q2_text.grid_forget()
        q3_text.grid_forget()
        q4_text.grid_forget()

    #Images
    image1 = Label(image=List_images[0])
    image2 = Label(image=processed_images[0])
    image3 = Label(image=List_images[1])
    image4 = Label(image=processed_images[1])
    
    image5 = Label(image=List_images[2])
    image6 = Label(image=processed_images[2])
    image7 = Label(image=List_images[3])
    image8 = Label(image=processed_images[3])

    #Texts
    q1_text = Label(root, text="Frame {}".format(begin_frame+begin), fg = "red")
    q2_text = Label(root, text="Frame {}".format(begin_frame+begin+frame_interval*1))
    q3_text = Label(root, text="Frame {}".format(begin_frame+begin+frame_interval*2))
    q4_text = Label(root, text="Frame {}".format(begin_frame+begin+frame_interval*3)) 
    avg_otzu_th_text = Label(root, text=" Recommended threshold: %.2f"%thres)
    #interval_otzu_th_text = Label(root, text=" Recommended threshold for this page: %.2f"%recommended_thresh_interval)
    total_pg_text = Label(root, text="Page: {}/{}".format(current_page, int(total_pages)))

    avg_otzu_th_text.grid(row=4, column=0, sticky = "w") 
    total_pg_text.grid(row=4, column=3, sticky = "e") 
    #interval_otzu_th_text.grid(row=4, column=1, sticky = "e")

    display_threshold()
    
    #ROW 1
    image1.grid(row=1, column=0) 
    image2.grid(row=1, column=1) 
    image3.grid(row=1, column=2)
    image4.grid(row=1, column=3)
    q1_text.grid(row=0, column=0, columnspan=2) 
    q2_text.grid(row=0, column=2, columnspan=2) 

    #ROW 2 
    image5.grid(row=3, column=0)
    image6.grid(row=3, column=1)
    image7.grid(row=3, column=2)
    image8.grid(row=3, column=3)
    q3_text.grid(row=2, column=0, columnspan=2)
    q4_text.grid(row=2, column=2, columnspan=2)
    #image_graph.grid(row=7,column=0,columnspan=6, rowspan=5, sticky="w") 

def draw_page_slider():
    #page slider
    global total_pages
    global page_slider
    page_slider = Scale(root,label="Select Page", orient="horizontal", length = 1295, resolution=1, from_=1, to=int(total_pages), tickinterval = 1)
    page_slider.set(1)
    page_slider.length = 1295 
    page_slider.sliderlength = 1295 
    page_slider.grid(row=5, column = 0, columnspan=4, sticky = "w")
    page_slider.configure(command=change_page)


def change_page(dummy):
    global button_back
    global button_forward
    global page_slider
    global current_page
    updated_p = page_slider.get() #updated
       
    if updated_p < current_page:
        #print("going back")
        for i in range(current_page - updated_p):
            button_back.invoke()

    elif updated_p > current_page:
        #print("going forward")
        for i in range(updated_p - current_page):
            button_forward.invoke()
        #forward button
   
def create_page_dict(total_pages):
    global page_threshold_dict
    global thres
    global frame_interval

    page_threshold_dict = {}
    #print("total_pages : ", total_pages)
    page_start_frame_number = 0
    page_end_frame_number = 3 * frame_interval
    for i in range(1, int(total_pages)+1):
        page_threshold_dict[i] = [False, thres, page_start_frame_number, page_end_frame_number]
        page_start_frame_number += 3 *frame_interval
        page_end_frame_number += 3 * frame_interval
    #print(page_threshold_dict)
    return

def create_frame_dict(begin, end):
    global frame_threshold_dict
    global thres
    #print("Creating frame dictionary")
    frame_threshold_dict = {}
    for i in range(begin, end):
        frame_threshold_dict[i] = thres
    return

def choose_input():
    global scale_label
    global input_dir
    global pattern
    global begin_frame
    global end_frame
    global images
    global scale_widget
    global button_forward
    global frame_interval
    global begin
    global end
    global thres
    global total_pages
    global current_page
    current_page = 1
    input_dir_dialog = filedialog.askdirectory(title='choose input directory', mustexist=True)
    if input_dir_dialog != None:
        #print(input_dir_dialog)
        input_dir = pathlib.Path(input_dir_dialog)
        input_dialog = InputDialog(input_dir)
        input_dialog.show()
        if input_dialog.was_ok_clicked():
            pattern, begin_frame, end_frame, images, frame_interval = input_dialog.get_input()
            if begin_frame < 0 or end_frame <0:
                print("ERROR begin/end frames cannot be less than 0")
            begin = 0
            #end = 12
            end = frame_interval * 3
            #print("frame_interval: ", frame_interval)
            button_forward.grid_forget()
            if end >= len(images): 
                button_forward = Button(root, text="Forward", command=forward, state=DISABLED) 
            else: 
                button_forward = Button(root, text="Forward", command=forward)
            button_forward.grid(row=7, column=2, sticky = "w")
            scale_widget.configure(command=display_segmented_image)
            setup_images(begin, end)
            update_binary_image(begin, end, scale_widget.get())
            thres  = calculate_otzu_threshold()
            #print("calculate_otzu+threshold(): ", thres)
            scale_widget.set(thres)
            total_pages = number_of_pages(begin_frame, end_frame, frame_interval)
            if total_pages > 50:
                
                messagebox.showwarning("Warning", "Frame interval is too small. Please pick a frame interval bigger than {} to ensure proper GUI formating".format( int((end_frame - begin_frame)/(3*45))) )
            #print("total_pages: ", total_pages)
            create_page_dict(total_pages)
            create_frame_dict(begin_frame, end_frame)
            draw_grid()
            draw_page_slider()
            display_segmented_image(scale_widget.get())

def print_frame_dict():
    global frame_threshold_dict
    global page_threshold_dict
    global frame_interval
    global total_pages
    # key = frame#/page#, val = threshold

    linear_smooth_threshold = page_threshold_dict[1][1]
    linear_smooth_func_of_frame_num = page_threshold_dict[1][1]
    for key, val in frame_threshold_dict.items():
        page_number = int(((key +1)/(frame_interval * 3)) + 1) #for this frame
        if page_number > total_pages:
            page_number = int(total_pages)
        if page_number != 1:
            previous_page_threshold = page_threshold_dict[page_number - 1][1]
            current_page_threshold  = page_threshold_dict[page_number][1]
            #print("currentpage: {} - previous page: {} = {}".format(current_page_threshold, previous_page_threshold, current_page_threshold - previous_page_threshold),end=" ")

            threshold_increase_per_frame = (current_page_threshold - previous_page_threshold)/(frame_interval * 3)

            #print("threshold_increase_per_frame: ", threshold_increase_per_frame, end = " ")

            linear_smooth_threshold += threshold_increase_per_frame

            frame_threshold_dict[key] = linear_smooth_threshold

            linear_smooth_func_of_frame_num = threshold_increase_per_frame * ((key+2) - page_threshold_dict[page_number -1][3]) + previous_page_threshold
        elif page_number == 1:
            frame_threshold_dict[key] = page_threshold_dict[1][1]
            
        #print("key:", key, "val:", val, "page#:", page_number, "(dict): page_threshold:",page_threshold_dict[page_number], "linear_smooth_threshold =", linear_smooth_threshold, "func_of_frame_num:", linear_smooth_func_of_frame_num)

    #print(frame_threshold_dict)

def choose_output():
    global scale_widget
    global frame_threshold_dict
    global page_threshold_dict
    global end_frame

    output_dir_dialog = filedialog.askdirectory(title='choose output directory', mustexist=True)

    if output_dir_dialog != None:
        #print("SAVING ---------- page threshold_dict: below")
        #print(page_threshold_dict)
        #print(output_dir_dialog)
        #print_frame_dict()
        output_dir = pathlib.Path(output_dir_dialog)
        processed_images = []
        #print("choose_output folder, len(Images): ", len(images))
        
        for i in range(end_frame):
            src = cv2.imread(str(images[i].resolve()), cv2.IMREAD_GRAYSCALE)
            thresh = scale_widget.get() # change this thresh depending on each image
            thresh = frame_threshold_dict[i]  
            #print("saving frame {}'s threshold {}".format(i, thresh))
            maxValue = 255 
            th, dst = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY_INV);      
        
            im_pil = Image.fromarray(dst)
            im_pil.save(output_dir / images[i].name)
    print("Successfully saved to {}".format(output_dir))


def background(func):
    global begin
    #print("from background function, begin is: ", begin)
    p = Process(target=func, args=(begin,))
    p.start()
    p.join()

#### INITIAL VIEW BEGINS -----------  
def initial_view():
    global scale_widget
    global inited
    global begin
    global end
    global root
    global button_exit
    global button_back
    global button_forward
    global button_set
    global frame_number
    global p1
    global scale_label
    # Window Setup 
    root = Tk()
    root.title("Binarize Images")
    root.geometry("1470x755")

    menu = Menu(root)
    root["menu"] = menu
    file_menu = Menu(menu, tearoff=0)
    menu.add_cascade(label='file', menu=file_menu)
    file_menu.add_command(label='open images', command=choose_input)
    file_menu.add_command(label='save segmented images', command=choose_output)

    global begin_frame
    global end_frame
    global images
    global pattern
    global List_images
    begin_frame = 0
    end_frame = 0
    images = []
    List_images = []
    pattern = ""
    scale_label = ""

    #scale widget init
    scale_widget = Scale(root,label="Select threshold", orient="horizontal", length = 1295, resolution=1, from_=0, to=254, tickinterval = 10)
    scale_widget.set(100)
    scale_widget.length = 1295
    scale_widget.sliderlength = 1295
    scale_widget.grid(row=6, column = 0, columnspan=4, sticky = "w")
    
    # We will have four button exit, back, forward, and save segmented images
    button_back = Button(root, text="Back", command=back, state=DISABLED)
    button_exit = Button(root, text="Exit", command=root.quit)
    button_forward = Button(root, text="Forward", command=forward)
    button_set = Button(root, text="Display Histogram", command=plot_a_graph)

    button_set.configure(foreground="red")
    
    # grid function is for placing the buttons in the frame
    button_exit.grid(row=7, column=1,sticky = "w")
    button_back.grid(row=7, column=1,sticky = "e")
    button_forward.grid(row=7, column=2,sticky = "w")
    button_set.grid(row=7, column=2, sticky = "e")
    #button_get_begin.grid(row=5, column=3)
    root.mainloop()

if __name__ == '__main__':
    global p1
    #p1 = Process(target=plot_a_graph)
    #p1.start()
    initial_view()
