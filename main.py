#  main.py
# Import the required libraries
from tkinter import *
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
from tkinter import filedialog
from tkinter import messagebox
from image_processor import ImageProcessor

image_processor = ImageProcessor()

def check_xlsx_file(full_excel_file):
    return os.path.exists(full_excel_file)


def file_ops(images_pd, index,excel_file):
    original_file_path = images_pd.at[index, 'File Name']
    file_name = os.path.basename(original_file_path)
    parent_directory = os.path.dirname(original_file_path)

    # Determine the classification prefix and the destination directory
    classification = images_pd.at[index, 'Classification']
    if classification == 'G':
        destination_dir = os.path.join(parent_directory, 'Accept')
    elif classification == 'B':
        destination_dir = os.path.join(parent_directory, 'Reject')
    else:
        destination_dir = parent_directory

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Rename and copy the original file by adding the classification prefix
    new_file_path = os.path.join(destination_dir, classification + file_name)
    print("Renaming and copying:", original_file_path, "to", new_file_path)

    # Perform the actual copy from the original file to the new destination
    shutil.copy(original_file_path, new_file_path)

    # Save the updated DataFrame back to the Excel file
    images_pd.to_excel(excel_file, index=False)

    # Return the updated DataFrame with the changed classifications
    return images_pd


def key_handler(key):

    print(key)
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
        parent_directory, last_directory = os.path.split(directory)
        last_directory = "d:/Image Data Files/" + last_directory + '.xlsx'
        # Check whether xlsx exists
        xlsx_exists = check_xlsx_file(last_directory)
        if xlsx_exists:
            return directory, last_directory
        else:
             return directory, False
    else:
        return None, None

def on_click(text, process_btn, view_btn, test_btn):
    x, y = open_file_dialog()
    dir_entry['text'] = x
    excel_entry['text'] = y
    if y:
        process_btn['state'] = 'normal'
        view_btn['state'] = 'normal'
        test_btn['state'] = 'normal'
    if x:
        process_btn['state'] = 'normal'
        view_btn['state'] = 'normal'
        test_btn['state'] = 'normal'


# Create the main window
window = Tk()
window.geometry("700x350")
window.title("Image Processing Application")

# Add a directory Label widget
dir_entry = Label(window, width=100)
dir_entry.pack()

# Add an Excel file Label widget
excel_entry = Label(window, width=100)
excel_entry.pack()

# Add Buttons in the window

# Create the button and set the command to call the process_images_and_create_dataframe() method
process_btn = tk.Button(
    window,
    text="Process Images",
    state=tk.DISABLED,
    command=lambda: image_processor.process_images(dir_entry['text'])
)
process_btn.pack()

view_btn = tk.Button(
    window,
    text="View Images",
    state=tk.DISABLED,
    command=lambda: image_processor.view_images(dir_entry['text'], excel_entry['text'], key_handler,file_ops)
)
view_btn.pack()

# Create a button for running the Train module
train_btn = tk.Button(window, text="Run Train", state=tk.DISABLED, command=lambda: run_random_forest_train(excel_entry['text']))
train_btn.pack()

# Create a button for running the Test module
test_btn = tk.Button(window, text="Run Test", state=tk.DISABLED, command=lambda: run_random_forest_test(excel_entry['text']))
test_btn.pack()


# Create a button for running the Select Directory module
select_dir_btn = tk.Button(window, text="Select Directory to Process", state=tk.NORMAL, command=lambda: on_click("A", process_btn, view_btn, test_btn))
select_dir_btn.pack()


# Create a button for running the Display Plots module
select_dir_btn = tk.Button(window, text="Display Plots", state=tk.NORMAL, command=lambda: image_processor.create_plots(excel_entry['text'], ['Brightness', 'Contour_Info', 'SHV', 'Harris_Corners', 'Laplacian']))
select_dir_btn.pack()





window.mainloop()