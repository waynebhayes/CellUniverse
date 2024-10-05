from tkinter import *
import tkinter
from tkinter.ttk import *
from tkinter import filedialog, messagebox

from math import sqrt

import json
import os

from image_canvas import ImageCanvasStatus

def calculate_growrate(start_count, end_count, num_ticks):
    if start_count <= 0 or end_count <= 0 or num_ticks <= 0:
        raise ValueError("All arguments must be positive integers.")
    
    ratio = end_count / start_count
    grow_rate = ratio ** (1 / num_ticks) - 1
    
    return grow_rate

class InputPanel(Frame):
    KWARGS = {
        "padding": 10
    }
    def __init__(self, root, **kwargs):
        # Merge default kwargs with user-supplied kwargs
        self.merged_kwargs = {**self.KWARGS, **kwargs}
        super().__init__(root, **self.merged_kwargs)
        
        # Create the label and Entry widget for max length input
        self.max_length_label = Label(self, text="Max length:")
        self.max_length_label.pack(side=TOP, padx=5, pady=5)
        self.max_length = Entry(self)
        self.max_length.pack(side=TOP, padx=5, pady=5)
        
        # Create the label and Entry widget for min length input
        self.min_length_label = Label(self, text="Min length:")
        self.min_length_label.pack(side=TOP, padx=5, pady=5)
        self.min_length = Entry(self)
        self.min_length.pack(side=TOP, padx=5, pady=5)
        
        # Create the label and Entry widget for max width input
        self.max_width_label = Label(self, text="Max width:")
        self.max_width_label.pack(side=TOP, padx=5, pady=5)
        self.max_width = Entry(self)
        self.max_width.pack(side=TOP, padx=5, pady=5)
        
        # Create the label and Entry widget for min width input
        self.min_width_label = Label(self, text="Min width:")
        self.min_width_label.pack(side=TOP, padx=5, pady=5)
        self.min_width = Entry(self)
        self.min_width.pack(side=TOP, padx=5, pady=5)

        # Rest of your code here ...
        # Create the label for max speed
        self.max_speed_label = Label(self, text="Max speed:")
        self.max_speed_label.pack(side=TOP, padx=5, pady=5)

        # Create the Entry widget for max speed input
        self.max_speed = Entry(self)
        self.max_speed.pack(side=TOP, padx=5, pady=5)

        # Create the label for max spin input
        self.max_spin_label = Label(self, text="Max spin:")
        self.max_spin_label.pack(side=TOP, padx=5, pady=5)

        # Create the Entry widget for max spin input
        self.max_spin = Entry(self)
        self.max_spin.pack(side=TOP, padx=5, pady=5)

        # Create the label for max growth
        self.max_growth_label = Label(self, text="Max growth:")
        self.max_growth_label.pack(side=TOP, padx=5, pady=5)

        # Create the Entry widget for max growth input
        self.max_growth = Entry(self)
        self.max_growth.pack(side=TOP, padx=5, pady=5)

        # Create the label for cell color
        self.cell_color_label = Label(self, text="Cell color:")
        self.cell_color_label.pack(side=TOP, padx=5, pady=5)

        # Create the Entry widget for cell color input
        self.cell_color_frame = Frame(self)
        self.cell_color = Entry(self.cell_color_frame, width=13)
        self.cell_color.grid(row=0, column=0)
        self.cell_color_button = Button(self.cell_color_frame, text="Pick", width=5, command=self.pick_cell_color)
        self.cell_color_button.grid(row=0, column=1)

        self.cell_color_frame.pack(side=TOP)
        # Create the label for Background color
        
        self.cell_color_label = Label(self, text="Background color:")
        self.cell_color_label.pack(side=TOP, padx=5, pady=5)

        # Create the Entry widget for background color input
        self.background_color_frame = Frame(self)
        self.background_color = Entry(self.background_color_frame, width=13)
        self.background_color.grid(row=0, column=0)
        self.background_color_button = Button(self.background_color_frame, text="Pick", width=5, command=self.pick_background_color)
        self.background_color_button.grid(row=0, column=1)

        self.background_color_frame.pack(side=TOP, padx=5, pady=5)

        self.split_rate_label = Label(self, text="Split rate:")
        self.split_rate_label.pack(side=TOP, padx=5, pady=5)

        # Create the label and Entry widget for split rate input
        self.split_rate_frame = Frame(self)
        self.split_rate = Entry(self.split_rate_frame, width=13)
        self.split_rate.grid(row=0, column=0)
        self.get_split_rate_button = Button(self.split_rate_frame, text="Get", width=5, command=self.get_split_rate)
        self.get_split_rate_button.grid(row=0, column=1)
        self.split_rate_frame.pack(side=TOP, padx=5, pady=5)

        # Create the button for generate config
        self.generate_button = Button(self, text="Generate", command=self.generate)
        self.generate_button.pack(side=TOP, padx=5, pady=5)

    def pick_cell_color(self):
        imageCanvasFrame1 = self.master.imageCanvasFrame1
        imageCanvasFrame2 = self.master.imageCanvasFrame2
        
        isSuccess = False
        isSuccess = imageCanvasFrame1.pick_cell_color() or isSuccess
        isSuccess = imageCanvasFrame2.pick_cell_color() or isSuccess

        if(not isSuccess):
            messagebox.showerror("Error", "Please first load at least one image before performing the operation.")

    def pick_background_color(self):
        imageCanvasFrame1 = self.master.imageCanvasFrame1
        imageCanvasFrame2 = self.master.imageCanvasFrame2
        
        isSuccess = False
        isSuccess = imageCanvasFrame1.pick_background_color() or isSuccess
        isSuccess = imageCanvasFrame2.pick_background_color() or isSuccess

        if(not isSuccess):
            messagebox.showerror("Error", "Please first load at least one image before performing the operation.")
    
    def get_split_rate(self):
        imageCanvasFrame1 = self.master.imageCanvasFrame1
        imageCanvasFrame2 = self.master.imageCanvasFrame2
        cell_count_1 = int(imageCanvasFrame1.cell_count_text_box.get())
        cell_count_2 = int(imageCanvasFrame2.cell_count_text_box.get())
        if cell_count_1 < cell_count_2:
            split_rate =  (cell_count_2 / cell_count_1) ** (1 / self.get_frame_difference()) - 1
        else:
            split_rate =  (cell_count_1 / cell_count_2) ** (1 / self.get_frame_difference()) - 1
        self.split_rate.delete(0, 'end')
        self.split_rate.insert(0, f'{split_rate:.2f}')

    def get_frame_difference(self):
        canvas1 = self.master.imageCanvasFrame1.imageCanvas
        canvas2 = self.master.imageCanvasFrame2.imageCanvas
        return abs(canvas1.curr_frame_idx - canvas2.curr_frame_idx)

    def generate(self):
        filename = filedialog.asksaveasfilename(defaultextension=".json",  title="Save configuration file", filetypes=(("JSON files", "*.json"),))
        template_filepath = f'{os.path.dirname(os.path.abspath(__file__))}/assets/config_template.json'
        with open(template_filepath, 'r') as f:
            config_template = json.load(f)

        syntheticImagePreviewPanel = self.master.syntheticImagePreviewPanel
        # Customize the template with the provided parameters
        config_template['bacilli.maxSpeed'] = float(self.max_speed.get())
        config_template['bacilli.maxSpin'] = float(self.max_spin.get())
        config_template['bacilli.maxGrowth'] = float(self.max_growth.get())
        config_template["bacilli.minWidth"] = float(self.min_width.get())
        config_template["bacilli.maxWidth"] = float(self.max_width.get())
        config_template["bacilli.minLength"] = float(self.min_length.get())
        config_template["bacilli.maxLength"] = float(self.max_length.get())
        config_template["simulation"]["background.color"] = float(self.background_color.get())
        config_template["simulation"]["cell.color"] = float(self.cell_color.get())
        config_template["simulation"]["cell.opacity"] = float(syntheticImagePreviewPanel.cell_opacity_entry.get())
        config_template["simulation"]["light.diffraction.sigma"] = float(syntheticImagePreviewPanel.diffraction_sigma_entry.get())
        config_template["simulation"]["light.diffraction.strength"] = float(syntheticImagePreviewPanel.diffraction_strength_entry.get())
        config_template["simulation"]["light.diffraction.truncate"] = float(syntheticImagePreviewPanel.diffraction_truncate_entry.get())
        config_template["prob.split"] = float(self.split_rate.get())
        
        # Write the customized configuration to file
        with open(filename, 'w+') as f:
            json.dump(config_template, f, indent = 4)
    