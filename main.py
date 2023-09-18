#  main.py

# Import the required libraries

from tkinter import Tk, Label, Button, Radiobutton, StringVar, Frame, Grid
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import shutil
import os
import numpy as np
import cv2
from PIL import Image, ImageTk
from image_functions import *
from file_functions_2 import *
from random_forest import *
from segment_class import *
from dominant import *
import matplotlib.pyplot as plt
import csv
import pandas as pd
import tkinter as tk
from image_processor import ImageProcessor
from data_handler import *

image_processor = ImageProcessor()
data_handler = DataHandler()

def check_xlsx_file(full_excel_file):
    return os.path.exists(full_excel_file)

def key_handler(key):

   #print(key)
    change_classification = False
    should_terminate = False
    # Check if key is a regular key event (ASCII value 0-255)
    if key >= 0 and key <= 255:
        print("Regular key pressed:", key)

    # If ESC key is pressed, break and save the DataFrame
    if key == 27:  # ASCII code for the "Esc" key
        print('Escape pressed')
        should_terminate= True  # Terminate the loop if "Esc" key is pressed

    # If End key is pressed, move to the last image
    elif key == 2293760:  # ASCII code for the "Home" key
        print('End pressed', key)
        change_classification = False  # Terminate the loop if "End" key is pressed

    # If home key is pressed, move to the first image
    elif key == 2359296:  # ASCII code for the "Home" key
        print('Home pressed', key)
        change_classification = False  # Terminate the loop if "Home" key is pressed

    # If up arrow key is pressed, move to the previous image
    elif key == 2490368:  # ASCII code for the "Up Arrow" key
        print('up arrow pressed', key)
        change_classification = False  # Terminate the loop if "<-" key is pressed

    # If down arrow key is pressed, move to the next image
    elif key == 2621440:  # ASCII code for the "Right Arrow" key
        change_classification = False  # Terminate the loop if "->" key is pressed

    # If 'g' or 'G' is pressed, change the classification to 'Accepted'
    elif key == (ord('g') or key == ord('G')):
        change_classification = True

    # If 'b' or 'B' is pressed, change the classification to 'Rejected'
    elif key == (ord('b') or key == ord('B')):
        change_classification = True


    return(change_classification, key)

def open_file_dialog():
    # Ask the user to select a directory or individual files
    directory = filedialog.askdirectory()
    fnames = []
    if directory:
        parent_directory, data_file_name = os.path.split(directory)
        excel_file_name = "d:/Image Data Files/" + data_file_name + '.xlsx'
        sql_file_name = "d:/Image Data Files sql/" + data_file_name + '.db'
        # Check whether xlsx exists
        xlsx_exists = check_xlsx_file(excel_file_name)
        sql_exists = check_xlsx_file(sql_file_name)
        if xlsx_exists and sql_exists:
            return directory, excel_file_name, sql_file_name
        elif xlsx_exists:
             return directory, excel_file_name, None
        elif sql_exists:
             return directory, None, sql_file_name
        else:
            return directory, None, None
    

    else:
        return None, None, None
    

def on_click(text, process_btn, view_btn, load_data_btn,excel_radio, sql_radio):
    dir_path, excel_path, sql_path = open_file_dialog()
    dir_entry['text'] = dir_path
    excel_entry['text'] = excel_path
    sql_entry['text'] = sql_path
    ImageProcessor.initialize(dir_path)
    if dir_path:
        process_btn['state'] = 'normal'
    if excel_path:
        load_data_btn['state'] = 'normal'
        view_btn['state'] = 'normal'
        excel_radio['state']= 'normal'

    if sql_path:
        load_data_btn['state'] = 'normal'
        view_btn['state'] = 'normal'
        sql_radio['state']= 'normal'

    original_radio['state'] = 'normal'
    brightness_radio['state'] = 'normal'
    contours_radio['state'] = 'normal'
    laplacian_radio['state'] = 'normal'



# Create the main window
window = Tk()
window.geometry("700x500")
window.title("Image Processing Application")

# Create a frame for the file/directory labels
frame_1 = Frame( window, bd=2, relief=tk.GROOVE)
frame_1.grid(row=0, column=0,pady=5, sticky="w") #, rowspan=3, columnspan=3

# Create a frame for the data source radio buttons
data_source_frame = Frame(window, bd=2, relief=tk.GROOVE)
data_source_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nw")

# Add a frame for the sort order radio buttons
sort_order_frame = Frame(window, bd=2, relief=tk.GROOVE)
sort_order_frame.grid(row=2, column=1, padx=5, pady=5, sticky="nw")


# Create a frame for the button group at the bottom
button_frame = Frame(window)
button_frame.grid(row=3, column=0, columnspan=3, pady=10,sticky="s")


# Frame_1 Contents


label_text = ["Directory:", "Excel File:","SQL File:"]
for counter, label in enumerate(label_text):
    Label(frame_1, text=label).grid(row=counter, column=0, sticky="w")

dir_entry = Label(frame_1, text="")
excel_entry = Label(frame_1, text="")
sql_entry = Label(frame_1, text="")

dir_entry.grid(row=0, column=1, sticky="w")
excel_entry.grid(row=1, column=1, sticky="w")
sql_entry.grid(row=2, column=1, sticky="w")



# data_source frame contents

# Add a label for data source
data_source_label = Label(data_source_frame, text="Data Source:")
data_source_label.grid(row=0, column=0, sticky="w")

# Create radio buttons for data source
source_var = StringVar()
excel_radio = Radiobutton(data_source_frame, text="Load Excel Data", state= tk.DISABLED, variable=source_var, value="excel")
sql_radio = Radiobutton(data_source_frame, text="Load SQL Data", state= tk.DISABLED, variable=source_var, value="sql")
excel_radio.grid(row=1, column=0, sticky="w")
sql_radio.grid(row=2, column=0, sticky="w")

# sort_order frame contents

# Add a label for sort order
sort_order_label = Label(sort_order_frame, text="Sort Order:")
sort_order_label.grid(row=0, column=0, sticky="w")

# Create radio buttons for sort order
sort_var = StringVar()
original_radio = Radiobutton(sort_order_frame, text="Original Sort", state=tk.DISABLED, variable=sort_var, value="original")
brightness_radio = Radiobutton(sort_order_frame, text="Sort by Brightness", state=tk.DISABLED, variable=sort_var, value="brightness")
laplacian_radio = Radiobutton(sort_order_frame, text="Sort by Laplacian", state=tk.DISABLED, variable=sort_var, value="laplacian")
contours_radio = Radiobutton(sort_order_frame, text="Sort by Contours", state=tk.DISABLED, variable=sort_var, value="contours")
original_radio.grid(row=1, column=0, sticky="w")
brightness_radio.grid(row=2, column=0, sticky="w")
laplacian_radio.grid(row=3, column=0, sticky="w")
contours_radio.grid(row=4, column=0, sticky="w")



# Button frame contents

# Create Buttons in the window

# Create a button for selecting the directory to process
select_dir_btn = tk.Button(
    button_frame,
    text="Select Directory",
    state=tk.NORMAL,
    command=lambda: on_click("A", process_btn, view_btn, load_data_btn, excel_radio, sql_radio)
)
select_dir_btn.grid(row=0, column=0)

# Create a button for processing images
process_btn = tk.Button(
    button_frame,
    text="Process Images",
    state=tk.DISABLED,
    command=lambda: image_processor.process_images(dir_entry['text'], excel_entry['text'], sql_entry['text'])
)
process_btn.grid(row=0, column=1)

# Create a button for viewing images
view_btn = tk.Button(
    button_frame,
    text="View Images",
    state=tk.DISABLED,
    command=lambda: image_processor.view_images(source_var.get(), sort_var.get(), key_handler)
)
view_btn.grid(row=0, column=2)

# Create a button for running the Train module
train_btn = tk.Button(
    button_frame,
    text="Run Train",
    state=tk.DISABLED,
    command=lambda: run_random_forest_train(excel_entry['text'])
)
train_btn.grid(row=0, column=3)

# Create a button for running the Test module
test_btn = tk.Button(
    button_frame,
    text="Run Test",
    state=tk.DISABLED,
    command=lambda: run_random_forest_test(excel_entry['text'])
)
test_btn.grid(row=0, column=4)

# Create a button for running the Load Data module
load_data_btn = tk.Button(
    button_frame,
    text="Load Data",
    state=tk.DISABLED,
    command=lambda: image_processor.load_data_from_data_handler(source_var.get())
)
load_data_btn.grid(row=0, column=5)

# Create a button for displaying plots
display_plots_btn = tk.Button(
    button_frame,
    text="Display Plots",
    state=tk.NORMAL,
    command=lambda: image_processor.create_plots(['Brightness', 'Contour_Info', 'SHV', 'Harris_Corners', 'Laplacian'])
)
display_plots_btn.grid(row=0, column=6)

# Create a button for displaying data
display_data_btn = tk.Button(
    button_frame,
    text="Display Data",
    state=tk.NORMAL,
    command=lambda: display_data(images_pd)
)
display_data_btn.grid(row=0, column=7)

window.mainloop()