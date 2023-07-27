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

def generate_statistics_image(row, resized_image, background_color=(0, 0, 0), text_color=(255, 255, 255),
                              text_size=0.5, text_thickness=1, text_position=(50, 50)):
    # Create the DataFrame for the statistics
    data = {'Attribute': [], 'Value': []}

    # Add the statistics to the DataFrame
    data['Attribute'].append('Image Statistics for directory:')
    data['Value'].append(os.path.dirname(row['File Name']))
    data['Attribute'].append('')
    data['Value'].append('')
    for column, value in row.items():
        if column == 'File Name':
            data['Attribute'].append('File Name:')
            data['Value'].append(os.path.basename(value))
        else:
            data['Attribute'].append(column + ':')
            data['Value'].append(str(value))

    # Create the DataFrame from the data
    df = pd.DataFrame(data)

    # Convert the DataFrame to a formatted table
    table_text = df.to_string(index=False, header=False, justify='left')

    # Split the table text into lines
    text_lines = table_text.split('\n')

    # Calculate the dimensions of the statistics image
    attribute_offset = 20  # Offset for the attribute column
    value_offset = 150  # Offset for the value column

    # Calculate the maximum text width
    max_text_width = max(
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)[0][0]
        for line in text_lines
    )

    # Calculate the overlay coordinates on the resized image
    overlay_start = (text_position[0], text_position[1])
    overlay_end = (overlay_start[0] + value_offset + max_text_width - 100,
                   overlay_start[1] + len(text_lines) * attribute_offset + 10)

    # Create the statistics image
    statistics_image = np.zeros(
        (overlay_end[1] - overlay_start[1], overlay_end[0] - overlay_start[0], 3), dtype=np.uint8)
    statistics_image[:] = background_color

    # Draw the text onto the statistics image
    for i, line in enumerate(text_lines):
        if i == 0:
            # Handle the title line separately
            title_position = (0, (i + 1) * attribute_offset)
            cv2.putText(statistics_image, line, title_position, cv2.FONT_HERSHEY_SIMPLEX,
                        text_size, text_color, text_thickness)
        else:
            parts = line.split(":", maxsplit=1)
            if len(parts) == 2:
                attribute, value = parts
            else:
                attribute = line.strip()
                value = ""  # Set a default value if needed

            # Strip the first 10 characters from the attribute and remove underscores
            attribute = attribute[10:].strip('_')

            attribute_position = (0, (i + 1) * attribute_offset)
            value_position = (value_offset, (i + 1) * attribute_offset)

            cv2.putText(statistics_image, attribute, attribute_position, cv2.FONT_HERSHEY_SIMPLEX,
                        text_size, text_color, text_thickness)
            cv2.putText(statistics_image, value, value_position, cv2.FONT_HERSHEY_SIMPLEX,
                        text_size, text_color, text_thickness)

    # Create a copy of the resized image
    overlay_image = resized_image.copy()

    # Overlay the statistics image onto the copy of the resized image
    overlay_image[overlay_start[1]:overlay_end[1], overlay_start[0]:overlay_end[0]] = statistics_image

    # Blend the overlay image with the resized image using transparency
    alpha = 0.9 # Adjust the transparency as needed
    blended_image = cv2.addWeighted(resized_image, 1 - alpha, overlay_image, alpha, 0)

    return blended_image



def load_image_for_display(image_path, image_orientation):
    image = cv2.imread(image_path)

    # Resize the image using the resize_image function
    if image_orientation == 'Landscape':
        resized_image, p = resize_image(image, ratio=30)  # Adjust the scale percent as desired
    if image_orientation == 'Portrait':
        resized_image, p = resize_image(image, ratio=20)  # Adjust the scale percent as desired
    print(resized_image.shape[:2])

    return resized_image



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
    command=lambda: image_processor.view_images(dir_entry['text'], excel_entry['text'], load_image_for_display, generate_statistics_image,key_handler,file_ops)
)
view_btn.pack()

# Create a button for running the Train module
train_btn = tk.Button(window, text="Run Train", state=tk.DISABLED, command=lambda: run_random_forest_train(excel_entry['text']))
train_btn.pack()

# Create a button for running the Test module
test_btn = tk.Button(window, text="Run Test", state=tk.DISABLED, command=lambda: run_random_forest_test(excel_entry['text']))
test_btn.pack()
select_dir_btn = tk.Button(window, text="Select Directory to Process", state=tk.NORMAL, command=lambda: on_click("A", process_btn, view_btn, test_btn))
select_dir_btn.pack()

window.mainloop()
