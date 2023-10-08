from tkinter import filedialog, ttk
from tkinter import *
from tkinter.ttk import *
from image_canvas import ImageCanvasFrame
from input_panel import InputPanel
from information_panel import InformationPanel
from synthetic_image_preview_panel import SyntheticImagePreviewPanel

import os
import re
class Launcher(Frame):
    KWARGS = {
        "padding": 10
    }
    def __init__(self, root, **kwargs):
        merged_kwargs = {**self.KWARGS, **kwargs}
        super().__init__(root, **merged_kwargs)
        self.label = Label(self, text="To start, select the folder containing the cell images.")
        self.label.pack()
        self.button = Button(self, text="Select Folder", command=self.launch)
        self.button.pack()

    def launch(self):
        folder_selected = filedialog.askdirectory()
        self.destroy()
        ConfigGenerator(self.master, folder_selected).pack()

class ConfigGenerator(Frame):
    KWARGS = {
        "padding": 10
    }
    @staticmethod
    def get_frame_files(folder_selected):
        frame_files = []
        pattern = re.compile(r"(frame(\d+)\.png)|(\d+\.png)|(\d+\.jpg)|frame-(\d+)\.jpg")  # Regular expression pattern to match "frameX.png" and "Y.jpg" files
        for file_name in os.listdir(folder_selected):
            match = pattern.match(file_name)
            if match:
                frame_files.append(file_name)
        
        # Sort the frame files based on the numeric part of the file name
        sorted_frame_files = sorted(frame_files, key=lambda x: int(re.search(r'\d+', x).group()))
        return sorted_frame_files
    
    def __init__(self, root, folder_selected, **kwargs):
        # Merge default kwargs with user-supplied kwargs
        merged_kwargs = {**self.KWARGS, **kwargs}
        super().__init__(root, **merged_kwargs)

        frame_files = ConfigGenerator.get_frame_files(folder_selected)

        self.imageCanvasFrame1 = ImageCanvasFrame(self, 1, folder_selected, frame_files)
        self.imageCanvasFrame1.grid(row=0, column=0)

        self.imageCanvasFrame2 = ImageCanvasFrame(self, 2, folder_selected, frame_files)
        self.imageCanvasFrame2.grid(row=0, column=1)
        
        self.informationPanel = InformationPanel(self)
        self.informationPanel.grid(row=0, column=2)
        
        self.syntheticImagePreviewPanel = SyntheticImagePreviewPanel(self)
        self.syntheticImagePreviewPanel.grid(row=0, column=3)

        self.input_panel = InputPanel(self)
        self.input_panel.grid(row=0, column=4)
        

def main():
    root = Tk()
    root.title("Config Generator")
    Launcher(root).pack()
    root.mainloop()


if __name__ == '__main__':
    main()
