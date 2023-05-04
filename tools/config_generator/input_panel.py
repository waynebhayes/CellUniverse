from tkinter import *
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

        # Create the first label for the left frame number input
        self.label1 = Label(self, text="Frame# of left:")
        self.label1.pack(side=TOP, padx=5, pady=5)

        # Create the first Entry widget for the left frame number input
        self.frame_number1 = Entry(self)
        self.frame_number1.pack(side=TOP, padx=5, pady=5)

        # Create the second label for the right frame number input
        self.label2 = Label(self, text="Frame# of right:")
        self.label2.pack(side=TOP, padx=5, pady=5)

        # Create the second Entry widget for the right frame number input
        self.frame_number2 = Entry(self)
        self.frame_number2.pack(side=TOP, padx=5, pady=5)

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

        # Number of cells of the left canvas
        self.cell_count_label1 = Label(self, text="Cell count(Left):")
        self.cell_count_label1.pack(side=TOP, padx=5, pady=5)

        self.cell_count_entry1 = Entry(self, width=20)
        self.cell_count_entry1.pack(side=TOP, padx=5, pady=5)

        # Number of cells of the right canvas
        self.cell_count_label2 = Label(self, text="Cell count(Right):")
        self.cell_count_label2.pack(side=TOP, padx=5, pady=5)

        self.cell_count_entry2 = Entry(self, width=20)
        self.cell_count_entry2.pack(side=TOP, padx=5, pady=5)
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
    def get_frame_difference(self):
        return int(self.frame_number2.get()) - int(self.frame_number1.get())

    def generate(self):
        output_folder = filedialog.askdirectory(title="Select output folder")
        template_filepath = f'{os.path.dirname(os.path.abspath(__file__))}/assets/config_template.json'
        with open(template_filepath, 'r') as f:
            config_template = json.load(f)

        min_max_cell_data = self.master.informationPanel.minMaxCellSizePanel.data
        syntheticImagePreviewPanel = self.master.syntheticImagePreviewPanel
        # Customize the template with the provided parameters
        config_template['bacilli.maxSpeed'] = float(self.max_speed.get())
        config_template['bacilli.maxSpin'] = float(self.max_spin.get())
        config_template['bacilli.maxGrowth'] = float(self.max_growth.get())
        config_template["bacilli.minWidth"] = min_max_cell_data['min_width']
        config_template["bacilli.maxWidth"] = min_max_cell_data['max_width']
        config_template["bacilli.minLength"] = min_max_cell_data['min_length']
        config_template["bacilli.maxLength"] = min_max_cell_data['max_length']
        config_template["simulation"]["background.color"] = float(self.background_color.get())
        config_template["simulation"]["cell.color"] = float(self.cell_color.get())
        config_template["simulation"]["cell.opacity"] = float(syntheticImagePreviewPanel.cell_opacity_entry.get())
        config_template["simulation"]["light.diffraction.sigma"] = float(syntheticImagePreviewPanel.diffraction_sigma_entry.get())
        config_template["simulation"]["light.diffraction.strength"] = float(syntheticImagePreviewPanel.diffraction_strength_entry.get())
        config_template["simulation"]["light.diffraction.truncate"] = float(syntheticImagePreviewPanel.diffraction_truncate_entry.get())
        
        # Write the customized configuration to file
        with open(f'{output_folder}/config.json', 'w') as f:
            json.dump(config_template, f, indent = 4)
    