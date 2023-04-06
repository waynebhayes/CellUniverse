from tkinter import ttk
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog, messagebox

from collections import defaultdict
from math import sqrt

import json
import os

class ConfigPanel(Frame):
    KWARGS = {
        "padding": 0
    }

    def __init__(self, root, **kwargs):
        # Merge default kwargs with user-supplied kwargs
        self.merged_kwargs = {**self.KWARGS, **kwargs}
        super().__init__(root, **self.merged_kwargs)

        self.selectedCellInfoPanel = SelectedCellInfoPanel(self)
        self.selectedCellInfoPanel.pack()

        self.minMaxCellSizePanel = MinMaxCellSizePanel(self)
        self.minMaxCellSizePanel.pack()

        self.inputPanel = InputPanel(self)
        self.inputPanel.pack()

class SelectedCellInfoPanel(Frame):
    KWARGS = {
        "padding": 0
    }

    def __init__(self, root, **kwargs):
        # Merge default kwargs with user-supplied kwargs
        self.merged_kwargs = {**self.KWARGS, **kwargs}
        super().__init__(root, **self.merged_kwargs)

        # Create the labels
        self.speed_label = Label(self, text="speed: ", takefocus=True)
        self.spin_label = Label(self, text="spin speed: ", takefocus=True)
        self.growth_label = Label(self, text="growth rate: ", takefocus=True)
        self.calculate_button = Button(self, text="calculate", command=self.calculate)

        # Pack the labels and button vertically
        self.speed_label.pack(side=TOP)
        self.spin_label.pack(side=TOP)
        self.growth_label.pack(side=TOP)
        self.calculate_button.pack(side=TOP)

    def set_labels(self, speed, spin, growth):
        # store a copy for other widgets to access
        self.data = {'speed': speed, 'spin': spin, 'growth': growth}

        self.speed_label.configure(text=f"speed: {speed:.2f}")
        self.spin_label.configure(text=f"spin speed: {spin:.2f}")
        self.growth_label.configure(text=f"growth rate: {growth:.2f}")

    def calculate(self):
        # obtain cell dicts from both ImageCanvas
        canvas1 = self.master.master.imageCanvasFrame1.imageCanvas
        canvas2 = self.master.master.imageCanvasFrame2.imageCanvas
        
        cell_dict1: defaultdict(dict) = canvas1.cell_dict
        cell_dict2: defaultdict(dict) = canvas2.cell_dict

        # if selected_id is not present, do nothing
        if canvas1.selected_id == None or canvas2.selected_id == None:
            return

        # obtain two selected cell
        cell_1 = cell_dict1[canvas1.selected_id]
        cell_2 = cell_dict2[canvas2.selected_id]

        frame_diff = self.master.inputPanel.get_frame_difference()
        # calculate speed
        p1 = cell_1['center']
        p2 = cell_2['center']

        distance_travelled = sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        speed = distance_travelled / frame_diff
        
        # calculate spin
        spin1 = cell_1['rotation']
        spin2 = cell_2['rotation']
        spin = abs(spin2- spin1) / frame_diff

        # calculate growth
        length1 = cell_1['length']
        length2 = cell_2['length']
        growth = (length2 - length1) / frame_diff

        self.set_labels(speed, spin, growth)
        
class InputPanel(Frame):
    KWARGS = {
        "padding": 0
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

        # Create the button for generate config
        self.generate_button = Button(self, text="Generate", command=self.generate)
        self.generate_button.pack(side=TOP, padx=5, pady=5)

    def get_frame_difference(self):
        return int(self.frame_number2.get()) - int(self.frame_number1.get())

    def generate(self):
        output_folder = filedialog.askdirectory(title="Select output folder")
        template_filepath = f'{os.path.dirname(os.path.abspath(__file__))}/config_template.json'
        with open(template_filepath, 'r') as f:
            config_template = json.load(f)

        min_max_cell_data = self.master.minMaxCellSizePanel.data
        # Customize the template with the provided parameters
        config_template['bacilli.maxSpeed'] = float(self.max_speed.get())
        config_template['bacilli.maxSpin'] = float(self.max_spin.get())
        config_template['bacilli.maxGrowth'] = float(self.max_growth.get())
        config_template["bacilli.minWidth"] = min_max_cell_data['min_width']
        config_template["bacilli.maxWidth"] = min_max_cell_data['max_width']
        config_template["bacilli.minLength"] = min_max_cell_data['min_length']
        config_template["bacilli.maxLength"] = min_max_cell_data['max_length']

        # Write the customized configuration to file
        with open(f'{output_folder}/config.json', 'w') as f:
            json.dump(config_template, f, indent = 4)
    
class MinMaxCellSizePanel(Frame):
    KWARGS = {
        "padding": 0
    }

    def __init__(self, root, **kwargs):
        # Merge default kwargs with user-supplied kwargs
        self.merged_kwargs = {**self.KWARGS, **kwargs}
        super().__init__(root, **self.merged_kwargs)

        # Create the labels and button
        self.min_cell_length_label = Label(self, text="Min cell length:")
        self.max_cell_length_label = Label(self, text="Max cell length:")
        self.min_cell_width_label = Label(self, text="Min cell width:")
        self.max_cell_width_label = Label(self, text="Max cell width:")
        self.update_button = Button(self, text="Update", command=self.update)

        # Pack the labels and button vertically
        self.min_cell_length_label.pack(side=TOP)
        self.max_cell_length_label.pack(side=TOP)
        self.min_cell_width_label.pack(side=TOP)
        self.max_cell_width_label.pack(side=TOP)
        self.update_button.pack(side=TOP)

    def set_labels(self, min_length, max_length, min_width, max_width):
        # store a copy for other widgets to access
        self.data = {'min_length': min_length, 'max_length': max_length, 'min_width': min_width, 'max_width': max_width}
        self.min_cell_length_label.configure(text=f"Min cell length: {min_length}")
        self.max_cell_length_label.configure(text=f"Max cell length: {max_length}")
        self.min_cell_width_label.configure(text=f"Min cell width: {min_width}")
        self.max_cell_width_label.configure(text=f"Max cell width: {max_width}")

    def update(self):
        # obtain cell dicts from both ImageCanvas
        canvas1 = self.master.master.imageCanvasFrame1.imageCanvas
        canvas2 = self.master.master.imageCanvasFrame2.imageCanvas
        
        cell_dict1: defaultdict(dict) = canvas1.cell_dict
        cell_dict2: defaultdict(dict) = canvas2.cell_dict

        # if no cells available, do nothing
        if len(cell_dict1) == 0 and len(cell_dict2) == 0:
            return
        
        # obtain all cells
        cells = []
        cells.extend(list(cell_dict1.values()))
        cells.extend(list(cell_dict2.values()))

        # obtain lengths, widths
        lengths = [cell['length'] for cell in cells]
        widths = [cell['width'] for cell in cells]

        min_length, max_length, min_width, max_width = min(lengths), max(lengths), min(widths), max(widths)

        self.set_labels(min_length, max_length, min_width, max_width)