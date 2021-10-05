import os
import re
import numpy as np
import pandas as pd
import tkinter as tk
import pydicom
import concurrent.futures
from scipy import ndimage
from tkinter import filedialog
from PIL import Image, ImageTk
from time import sleep


TITLE = 'RSNA Data Viewer'
RESOLUTION = "1920x1080"
RESIZE_WIDTH = 192
RESIZE_HEIGHT = 192
RESIZE_DEPTH = 192
SCAN_TYPES = ['FLAIR', 'T1w', 'T1wCE', 'T2w']


def sort_by_filename(list_: list) -> list:
    """
    Sorts list by filename in ascending order
       Before                After
    'Image-1.dcm',        'Image-1.dcm',
    'Image-10.dcm',       'Image-2.dcm',
    'Image-100.dcm',      'Image-3.dcm',
    'Image-101.dcm',      'Image-4.dcm',
    'Image-102.dcm',      'Image-5.dcm',
     ...                   ...
    """
    return sorted(list_, key=lambda f: int(re.sub('\D', '', f)))


def normalize(image: np.ndarray) -> np.ndarray:
    image = image - np.min(image)
    if np.max(image) != 0:
        image = image / np.max(image)
    image = (image * 255).astype(np.uint8)
    return image


def resize(image: np.ndarray) -> np.ndarray:
    image = np.array(image)
    cur_w, cur_h, cur_d = image.shape
    w = cur_w / RESIZE_WIDTH
    h = cur_h / RESIZE_HEIGHT
    d = cur_d / RESIZE_DEPTH
    factor_w = 1 / w
    factor_h = 1 / h
    factor_d = 1 / d
    factors = (factor_w, factor_h, factor_d)
    image = ndimage.zoom(image, factors, order=1)
    return image


def load_image(path: str) -> np.array:
    image = pydicom.read_file(path)
    return normalize(image.pixel_array)


class Viewer():

    def __init__(self) -> None:
        # Tkinter initialization
        self.root = tk.Tk()
        self.root.title(TITLE)
        self.root.geometry(RESOLUTION)
        # Variables
        self.current_file = ''
        self.path = ''
        self.folder = []
        self.scans = []
        self.files = []
        self.df = pd.read_csv('train_labels.csv')
        # Ratios, % of main window
        self.SELECTION_HEIGHT = 0.1
        self.CLASSES_HEIGHT = 0.05 # SUM 0.15
        self.IMAGES_HEIGHT = 0.5 # SUM 0.65
        self.META_HEIGHT = 0.2 # SUM 0.85
        self.SLIDER_HEIGHT = 0.1 # SUM 0.95
        # Start
        self.build_window()
        self.root.mainloop()

    def build_window(self) -> None:
        """Function that builds a window. Almost all GUI stuff is happening here"""
        sums = [self.SELECTION_HEIGHT, self.CLASSES_HEIGHT, self.IMAGES_HEIGHT, self.META_HEIGHT, self.SLIDER_HEIGHT]
        assert sum(sums) <= 1, 'Must be less than 100% (1.)'
        # Selection Frame
        self.selection_frame = tk.Frame(self.root)
        self.selection_frame.place(relwidth=1, relheight=self.SELECTION_HEIGHT)
        self.log_area = tk.Text(self.selection_frame)
        self.log_area.pack(side=tk.LEFT, fill=tk.BOTH, padx=15)
        self.update_log_area('Log area.')
        self.file_label = tk.Label(self.selection_frame, font=("Courier", 30))
        self.file_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
        self.change_text(self.file_label, "Select datasets (train, test)\nor a specific case (not implemented)")
        # Buttons Frame, inside Selection
        self.buttons_frame = tk.Frame(self.selection_frame)
        self.buttons_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=15, pady=10)
        self.go_left = tk.Button(self.buttons_frame, text='<', font=("Courier", 35), command=self.previous_case)
        self.go_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
        self.change_state(self.go_left, "disabled")
        self.open_dir = tk.Button(self.buttons_frame, text='Open', font=("Courier", 25), command=self.open)
        self.open_dir.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
        self.go_right = tk.Button(self.buttons_frame, text='>', font=("Courier", 35), command=self.next_case)
        self.go_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
        self.change_state(self.go_right, "disabled")
        # Class labels
        self.class_frame = tk.Frame(self.root)
        self.class_frame.place(rely=self.SELECTION_HEIGHT+0.01, relwidth=1, relheight=self.CLASSES_HEIGHT)
        self.class_labels = [tk.Label(self.class_frame, font=("Courier", 20)) for _ in range(4)]
        for i, label in enumerate(self.class_labels):
            self.change_text(label, f"{SCAN_TYPES[i]}")
            label.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
        # Image Canvas
        self.images_frame = tk.Frame(self.root)#, bg='blue')
        self.images_frame.place(relx=0, rely=self.SELECTION_HEIGHT+self.CLASSES_HEIGHT+0.01, relwidth=1, relheight=self.IMAGES_HEIGHT)
        self.image_canvases = [tk.Canvas(self.images_frame) for _ in range(4)]
        for ic in self.image_canvases:
            ic.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
        # Meta info
        self.meta_frame = tk.Frame(self.root)#, bg='blue')
        self.meta_frame.place(relx=0, rely=self.SELECTION_HEIGHT+self.CLASSES_HEIGHT+self.IMAGES_HEIGHT+0.01, relwidth=1, relheight=self.META_HEIGHT)
        self.meta_labels = [tk.Label(self.meta_frame, font=("Courier", 13), anchor='w', text='bla\nbla\nbla\nbla') for _ in range(4)]
        for label in self.meta_labels:
            label.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES, padx=15)
        # Slider
        self.update_slider()
        self.change_state(self.slider, "disabled")

    def update_slider(self) -> None:
        """Updates length of the slider to correspond to max scan depth"""
        if self.scans:
            to = max([len(x) for x in self.scans])
        else:
            to = 1
        self.slider = tk.Scale(self.root, from_=1, to=to, tickinterval=10, orient=tk.HORIZONTAL, command=self.update_images)
        self.slider.place(relx=0.05, rely=0.9, relwidth=0.88, relheight=self.SLIDER_HEIGHT)

    def update_log_area(self, text: str) -> None:
        """Log area updater. Pass text to insert"""
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, text+'\n')
        self.log_area.update()
        self.log_area.see('end')
        self.log_area.config(state=tk.DISABLED)

    def previous_case(self) -> None:
        """Loads previous case to memory"""
        if self.selected_i-1 > 0:
            self.selected_i -= 1
            self.load_case()
            self.change_state(self.go_right, "active")
        else:
            self.change_state(self.go_left, "disabled")

    def next_case(self) -> None:
        """Loads next case to memory"""
        if self.selected_i+1 < len(self.folder):
            self.selected_i += 1
            self.load_case()
            self.change_state(self.go_left, "active")
        else:
            self.change_state(self.go_right, "disabled")

    def change_state(self, element: object, state: str) -> None:
        """Change state of an element and update it"""
        element.config(state=state, takefocus=0)
        element.update()

    def change_text(self, label: object, text: str) -> None:
        """Update text of a label and update it"""
        label.configure(text=text)
        label.update()

    def update_title(self) -> None:
        """Update upper label with "Case: , MGMT_value"" """
        if 'test' not in self.current_path:
            prediction = self.df[self.df['BraTS21ID'] == int(self.current_file)].iloc[0]['MGMT_value']
        else:
            prediction = 'Unknown (test)'
        self.change_text(self.file_label, f"Case: {self.current_file}\nMGMT_value: {prediction}")

    def reset_classes(self) -> None:
        """Resets all class labels"""
        for label in self.class_labels:
            self.change_text(label, "")

    def reset_images(self) -> None:
        """Tries to reset all images. For some reason, it's not working."""
        for ic in self.image_canvases:
            image = Image.fromarray(np.zeros((1, 1)))
            image = ImageTk.PhotoImage(image)
            ic.create_image(256, 256, anchor="center", image=image)
            ic.update()

    def load_case(self) -> None:
        """Loads new case to memory"""
        self.change_state(self.slider, "disabled")
        self.current_path = f'{self.path}/{self.folder[self.selected_i]}'
        self.current_file = self.current_path.split('/')[-1]
        self.change_text(self.file_label, f"Loading {self.current_file}...")
        self.change_state(self.open_dir, "disabled")
        self.change_state(self.go_left, "disabled")
        self.change_state(self.go_right, "disabled")
        self.files = {
            "FLAIR": sort_by_filename(os.listdir(self.current_path+'/FLAIR')),
            "T1w": sort_by_filename(os.listdir(self.current_path+'/T1w')),
            "T1wCE": sort_by_filename(os.listdir(self.current_path+'/T1wCE')),
            "T2w": sort_by_filename(os.listdir(self.current_path+'/T2w'))
        }
        self.load_scans()
        self.change_state(self.go_right, "active")
        self.change_state(self.open_dir, "active")
        self.change_state(self.slider, "normal")

    def open(self) -> None:
        """Opens new datasets and allows buttons (< >) to work"""
        path = filedialog.askdirectory()
        if type(path) == str:
            if os.path.isdir(path):
                self.path = path
                self.update_log_area(f'Selected {self.path}')
                self.folder = os.listdir(self.path)
                self.selected_i = 0
                self.load_case()
            else:
                return

    def load_scans(self) -> None:
        """Loads all scans of case to memory"""
        # self.reset_classes()
        self.reset_images()
        self.update_log_area(f'Loading case {self.current_file}')
        pre_resized = []
        for i, (type_, content) in enumerate(self.files.items()):
            self.update_log_area(f'Loading {type_}...')
            paths = [f"{self.current_path}/{type_}/{name}" for name in content]
            with concurrent.futures.ProcessPoolExecutor() as executor:
                slices = list(executor.map(load_image, paths))
            pre_resized.append(slices)
        self.update_log_area(f'Resizing scans to {RESIZE_DEPTH}x{RESIZE_WIDTH}x{RESIZE_HEIGHT}...')
        with concurrent.futures.ProcessPoolExecutor() as executor:
            self.scans = list(executor.map(resize, pre_resized))
        self.update_log_area('Done.')
        self.update_slider()
        self.update_images(1)
        self.update_title()

    def update_images(self, depth: int) -> None:
        """Updates slice of scans on slider event"""
        self.current_images = []
        depth = int(depth)-1
        for i, ic in enumerate(self.image_canvases):
            bound = min(len(self.scans[i])-1, depth)
            image = Image.fromarray(self.scans[i][bound])
            image = ImageTk.PhotoImage(image)
            self.current_images.append(image)
            # yes, 'cur_images[i]' is silly, but you have to pin images in memory
            # otherwise, python gc will delete them. Try to comment the lines and
            # see for yourself
            ic.create_image(256, 256, anchor="center", image=self.current_images[i])


if __name__ == '__main__':
    app = Viewer()





# # Maybe later
# class Resizing_Image(tk.Frame):
#     def __init__(self, master, *pargs):
#         Frame.__init__(self, master, *pargs)
#         self.image = Image.open("./resource/Background.gif")
#         self.img_copy= self.image.copy()
#         self.background_image = ImageTk.PhotoImage(self.image)
#         self.background = Label(self, image=self.background_image)
#         self.background.pack(fill=BOTH, expand=YES)
#         self.background.bind('<Configure>', self._resize_image)

#     def _resize_image(self,event):
#         new_width = event.width
#         new_height = event.height
#         self.image = self.img_copy.resize((new_width, new_height))
#         self.background_image = ImageTk.PhotoImage(self.image)
#         self.background.configure(image = self.background_image)