from tkinter import *
from tkinter.ttk import *
from collections import defaultdict
from math import sqrt

class InformationPanel(Frame):
    KWARGS = {
        "padding": 10,
    }

    def __init__(self, root, **kwargs):
        # Merge default kwargs with user-supplied kwargs
        self.merged_kwargs = {**self.KWARGS, **kwargs}
        super().__init__(root, **self.merged_kwargs)

        self.selectedCellInfoPanel = SelectedCellInfoPanel(self)
        self.selectedCellInfoPanel.pack()

        self.minMaxCellSizePanel = MinMaxCellSizePanel(self)
        self.minMaxCellSizePanel.pack()

        self.colorInfoPanel = ColorInfoPanel(self)
        self.colorInfoPanel.pack()

class SelectedCellInfoPanel(Frame):
    KWARGS = {
        "padding": 10
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
        self.speed_label.pack(side=TOP, anchor=W)
        self.spin_label.pack(side=TOP)
        self.growth_label.pack(side=TOP)
        self.calculate_button.pack(side=TOP)

    def set_labels(self, speed, spin, growth):
        # store a copy for other widgets to access
        self.data = {'speed': speed, 'spin': spin, 'growth': growth}

        self.speed_label.configure(text=f"speed: {speed:.3f}")
        self.spin_label.configure(text=f"spin speed: {spin:.3f}")
        self.growth_label.configure(text=f"growth rate: {growth:.3f}")

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

        frame_diff = self.master.master.input_panel.get_frame_difference()
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
        

class MinMaxCellSizePanel(Frame):
    KWARGS = {
        "padding": 10
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

class ColorInfoPanel(Frame):
    KWARGS = {
        "padding": 10
    }

    def __init__(self, root, **kwargs):
        # Merge default kwargs with user-supplied kwargs
        self.merged_kwargs = {**self.KWARGS, **kwargs}
        super().__init__(root, **self.merged_kwargs)

        # Create the cell color line
        cell_color_frame = Frame(self)
        cell_color_label = Label(cell_color_frame, text="Cell Color:")
        cell_color_label.pack(side=LEFT)

        cell_color_cube = Canvas(cell_color_frame, width=20, height=20, bg='white')
        cell_color_cube.pack(side=LEFT)

        cell_color_frame.pack(side=TOP)

        # Create the cell color line
        background_color_frame = Frame(self)
        background_color_label = Label(background_color_frame, text="Back Color:")
        background_color_label.pack(side=LEFT)

        background_color_cube = Canvas(background_color_frame, width=20, height=20, bg='white')
        background_color_cube.pack(side=LEFT)

        background_color_frame.pack(side=TOP)

        # Save a reference to cell_color_cube and background color_cube for future use
        self.cell_color_cube = cell_color_cube
        self.background_color_cube = background_color_cube
        
    def set_cell_color(self, gray_value):
        hex_value = f"#{int(gray_value*255):02x}{int(gray_value*255):02x}{int(gray_value*255):02x}"
        self.cell_color_cube.configure(bg=hex_value)

    def set_background_color(self, gray_value):
        hex_value = f"#{int(gray_value*255):02x}{int(gray_value*255):02x}{int(gray_value*255):02x}"
        self.background_color_cube.configure(bg=hex_value)

