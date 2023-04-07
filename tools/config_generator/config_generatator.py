from tkinter import ttk
from tkinter import *
from tkinter.ttk import *
from image_canvas import ImageCanvasFrame
from config_panel import ConfigPanel


class ConfigGenerator(Frame):
    KWARGS = {
        "padding": 10
    }
    def __init__(self, root, **kwargs):
        # Merge default kwargs with user-supplied kwargs
        merged_kwargs = {**self.KWARGS, **kwargs}
        super().__init__(root, **merged_kwargs)

        self.imageCanvasFrame1 = ImageCanvasFrame(self)
        self.imageCanvasFrame1.grid(row=0, column=0)

        self.imageCanvasFrame2 = ImageCanvasFrame(self)
        self.imageCanvasFrame2.grid(row=0, column=1)
        
        self.configPanel = ConfigPanel(self)
        self.configPanel.grid(row=0, column=2)
        

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