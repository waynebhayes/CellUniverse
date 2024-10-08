from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import os
import sys
import pandas as pd
import PIL
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/../../src')
from cell import Bacilli
from optimization import generate_synthetic_image
from global_optimization.Modules import LineageM


class SyntheticImagePreviewPanel(Frame):
    KWARGS = {
        "padding": 10,
    }

    def __init__(self, root, **kwargs):
        # Merge default kwargs with user-supplied kwargs
        self.merged_kwargs = {**self.KWARGS, **kwargs}
        super().__init__(root, **self.merged_kwargs)

        self.title = Label(self, text="Synthetic Preview")
        self.title.pack()

        # Create a Canvas widget and pack it into the frame
        self.canvas = Canvas(self, width=120, height=200, bg='white')
        self.canvas.pack()

        # Create labels and input boxes for the cell parameters
        self.cell_length_label = Label(self, text="cell length")
        self.cell_length_label.pack()
        self.cell_length_entry = Entry(self)
        self.cell_length_entry.insert(END, "49")
        self.cell_length_entry.pack(padx=5, pady=5)

        self.cell_width_label = Label(self, text="cell width")
        self.cell_width_label.pack()
        self.cell_width_entry = Entry(self)
        self.cell_width_entry.insert(END, "10")
        self.cell_width_entry.pack(padx=5, pady=5)

        self.rotation_label = Label(self, text="rotation")
        self.rotation_label.pack()
        self.rotation_entry = Entry(self)
        self.rotation_entry.insert(END, "1.43")
        self.rotation_entry.pack(padx=5, pady=5)



        self.cell_opacity_label = Label(self, text="cell opacity")
        self.cell_opacity_label.pack()
        self.cell_opacity_entry = Entry(self)
        self.cell_opacity_entry.insert(END, "0.2")
        self.cell_opacity_entry.pack(padx=5, pady=5)

        # Create labels and input boxes for the three diffraction parameters
        self.diffraction_sigma_label = Label(self, text="diffraction sigma")
        self.diffraction_sigma_label.pack()
        self.diffraction_sigma_entry = Entry(self)
        self.diffraction_sigma_entry.insert(END, "9.0")
        self.diffraction_sigma_entry.pack(padx=5, pady=5)

        self.diffraction_strength_label = Label(self, text="diffraction strength")
        self.diffraction_strength_label.pack()
        self.diffraction_strength_entry = Entry(self)
        self.diffraction_strength_entry.insert(END, "0.4")
        self.diffraction_strength_entry.pack(padx=5, pady=5)

        self.diffraction_truncate_label = Label(self, text="diffraction truncate")
        self.diffraction_truncate_label.pack()
        self.diffraction_truncate_entry = Entry(self)
        self.diffraction_truncate_entry.insert(END, "1")
        self.diffraction_truncate_entry.pack(padx=5, pady=5)

        # Create a "Generate" button
        self.generate_button = Button(self, text="Generate", command=self.generate_image)
        self.generate_button.pack()

    def generate_image(self):
        # check if cell color and background color are provided
        cell_color = self.master.input_panel.cell_color.get()
        background_color = self.master.input_panel.background_color.get()
        if(cell_color == '' or background_color == ''):
            messagebox.showerror("Error", f"Please fill in cell color and background color in the input panel.")
            return
        
        zoom_factor = self.master.imageCanvasFrame1.imageCanvas.zoom_factor

        simulation_config = {   
            'background.color': float(background_color),
            'cell.color': float(cell_color),
            'cell.opacity': float(self.cell_opacity_entry.get()),
            'image.type': 'graySynthetic',
            'light.diffraction.sigma': float(self.diffraction_sigma_entry.get()),
            'light.diffraction.strength': float(self.diffraction_strength_entry.get()),
            'light.diffraction.truncate': int(self.diffraction_truncate_entry.get()),
            'padding': 0
        }

        # synthimage_shape is of format (height, width) due to how the function generate_synthetic_image implements.
        synthimage_shape = (int(200 * zoom_factor), int(120 * zoom_factor))

        lineage = LineageM(simulation_config)
        new_cell = Bacilli("b00", synthimage_shape[1] / 2, synthimage_shape[0] / 2, float(self.cell_width_entry.get()), 
                            float(self.cell_length_entry.get()), float(self.rotation_entry.get()), None, 
                            float(self.cell_opacity_entry.get()))
        lineage.frames[-1].add_cell(new_cell)

        synthimage, cellmap = generate_synthetic_image(lineage.frames[0].nodes, synthimage_shape, simulation_config)

        # Display the synthetic image on the canvas
        synthimage_int8 = (synthimage * 255).astype(np.uint8)
        img = PIL.Image.fromarray(synthimage_int8, mode='L')
        img = img.resize((120, 200))
        img_tk = PIL.ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=img_tk, anchor=NW)
        self.canvas.image = img_tk
