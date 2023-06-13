from tkinter import ttk
from tkinter import *
from tkinter.ttk import *
from image_canvas import ImageCanvasFrame
from input_panel import InputPanel
from information_panel import InformationPanel
from synthetic_image_preview_panel import SyntheticImagePreviewPanel

class ConfigGenerator(Frame):
    KWARGS = {
        "padding": 10
    }
    def __init__(self, root, **kwargs):
        # Merge default kwargs with user-supplied kwargs
        merged_kwargs = {**self.KWARGS, **kwargs}
        super().__init__(root, **merged_kwargs)

        self.imageCanvasFrame1 = ImageCanvasFrame(self, 1)
        self.imageCanvasFrame1.grid(row=0, column=0)

        self.imageCanvasFrame2 = ImageCanvasFrame(self, 2)
        self.imageCanvasFrame2.grid(row=0, column=1)
        
        self.informationPanel = InformationPanel(self)
        self.informationPanel.grid(row=0, column=2)
        
        self.syntheticImagePreviewPanel = SyntheticImagePreviewPanel(self)
        self.syntheticImagePreviewPanel.grid(row=0, column=3)

        self.input_panel = InputPanel(self)
        self.input_panel.grid(row=0, column=4)
        

def main():
    # WINDOW_WIDTH, WINDOW_HEIGHT = 900, 640
    # WINDOW_SIZE = f'{WINDOW_WIDTH}x{WINDOW_HEIGHT}'

    root = Tk()
    root.title("Config Generator")
    # root.geometry(WINDOW_SIZE)

    ConfigGenerator(root).pack()
    root.mainloop()

if __name__ == '__main__':
    main()