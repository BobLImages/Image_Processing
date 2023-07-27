import pandas as pd
from file_functions_2 import *
from tkinter import messagebox




class ImageProcessor:
    def __init__(self):
        self.images_pd = pd.DataFrame()

    def load_data_from_excel(self, excel_file):
        self.images_pd = pd.read_excel(excel_file)
        return self.images_pd

    def save_dataframe_to_excel(self):
        overwrite = False
        save_data = False

        excel_file = "d:/Image Data Files/" + os.path.basename(os.path.dirname(self.images_pd['File Name'][0])) + '.xlsx'

        if os.path.exists(excel_file):
            # Prompt the user with a message box to confirm overwriting
            overwrite = messagebox.askyesno("File Exists", "The Excel file already exists. Do you want to overwrite it?")
            if not overwrite:
                print("Operation canceled. Existing file will not be overwritten.")
                return
        else:
            # Prompt the user with a message box to confirm saving the data
            save_data = messagebox.askyesno("File Does Not Exist", "The Excel file does not exist. Do you want to save the data?")
            if not save_data:
                print("Data not saved.")
                return

        if overwrite or save_data:
            self.images_pd.to_excel(excel_file, index=False)
            if save_data:
                excel_entry.config(text=excel_file)
                b3['state'] = 'normal'
                b3.pack()
        return self.images_pd


    def create_dataframe(self, images, dir_path):
        # Your image processing code here
        scale = 20
        enhanced_actions = False
        batch_size = 25

        self.images_pd = pd.DataFrame(
            columns=['File Name', 'Orientation', 'Brightness', 'Contrast', 'Haze Factor', 'Hough Info',
                     'Hough Circles', 'Harris Corners', 'Contour Info', 'Laplacian', 'SHV', 'Variance',
                     'Exposure', 'F-Stop', 'ISO', 'Black Pixels', 'Mid-tone Pixels', 'White Pixels',
                     'Faces', 'Eyes', 'Bodies', 'Focal Length', 'Classification'])

        # Populate the DataFrame with the data
        for image in images:
            self.images_pd = pd.concat([self.images_pd, pd.DataFrame({
                'File Name': [image.fname],
                'Orientation': [image.orientation],
                'Brightness': [round(image.brightness)],
                'Contrast': [round(image.contrast)],
                'Haze Factor': [round(image.haze_factor, 2)],
                'Hough Info': [image.hough_info[0]],
                'Hough Circles': [image.hough_circles],
                'Harris Corners': [image.harris_corners],
                'Contour Info': [image.contour_info[0]],
                'Laplacian': [round(image.laplacian)],
                'SHV': [round(image.shv, 2)],
                'Variance': [image.variance],
                'Exposure': [image.exposure],
                'F-Stop': [image.fstop],
                'ISO': [image.iso],
                'Black Pixels': [image.b_w[0]],
                'Mid-tone Pixels': [image.b_w[1]],
                'White Pixels': [image.b_w[2]],
                'Faces': [image.faces],
                'Eyes': [image.eyes],
                'Bodies': [image.bodies],
                'Focal Length': [image.focal_length],
                'Classification': [image.classification]
            })], ignore_index=True)


    def process_images(self, dir_path):
        # Your image processing code here
        scale = 20
        enhanced_actions = False
        batch_size = 25

        fnames = get_files(dir_path)
        images = load_images_batch(fnames, scale, enhanced_actions, batch_size)
        self.create_dataframe(images,dir_path)
        self.save_dataframe_to_excel()



    def view_images(self, dir_path, excel_file, load_image_for_display, generate_statistics_image,
                    key_handler, file_ops):
        images_pd = self.load_data_from_excel(excel_file)

        # Set rows and columns for the image grid
        rows = 5
        columns = 7

        landscape_data = create_grids(1344, 2018, rows, columns)
        portrait_data = create_grids(1344, 896, columns, rows)

        accepted_dir = os.path.join(dir_path, 'Accept')
        rejected_dir = os.path.join(dir_path, 'Reject')
        os.makedirs(accepted_dir, exist_ok=True)
        os.makedirs(rejected_dir, exist_ok=True)

        record_index = 0
        num_images = len(images_pd)
        print(num_images)

        while record_index < num_images:
            row = self.images_pd.iloc[record_index]
            print('Outer Loop:', record_index)
            print(row['File Name'], row['Orientation'])

            resized_image = load_image_for_display(row['File Name'], row['Orientation'])

            # Draw the rectangles on the resized image based on the image orientation
            if row['Orientation'] == 'Landscape':
                NW = landscape_data[0]
                zone_width = landscape_data[1]
                zone_height = landscape_data[2]
            else:
                NW = portrait_data[0]
                zone_width = portrait_data[1]
                zone_height = portrait_data[2]

            for nw_point in NW:
                # Calculate the coordinates of the top-left and bottom-right corners of the rectangle
                rectangle_start = nw_point
                rectangle_end = (nw_point[0] + zone_height, nw_point[1] + zone_width)
                cv2.rectangle(resized_image, rectangle_start, rectangle_end, (255, 255, 255), 2)

            resized_image = generate_statistics_image(row, resized_image, background_color=(0, 0, 0),
                                                      text_color=(0, 255, 255),
                                                      text_size=0.5, text_thickness=1, text_position=(10, 10))

            # Show the modified image
            converted_image = cv2.convertScaleAbs(resized_image)
            cv2.imshow("Modified Image", converted_image)

            change_classification = False  # Initialize the flag for each image
            should_terminate = False

            while True:
                print('Waiting...')
                # Wait for a key event
                key = cv2.waitKeyEx(0)

                change_classification, key = key_handler(key)

                print(key)
                if key == 2621440:
                    new_record_index = record_index + 1
                if key == 2490368:
                    new_record_index = record_index - 1
                elif key == 2293760:
                    new_record_index = num_images - 1
                elif key == 2359296:
                    new_record_index = 0
                elif key == 27:
                    self.images_pd.to_excel(excel_file, index=False)
                    cv2.destroyAllWindows()
                    return 
                else:
                    new_record_index = record_index + 1

                # If change_classification is True, perform file operations and reset the flag
                if change_classification and new_record_index != num_images:
                    print('cc =:', change_classification, new_record_index, num_images, key)
                    if key in (ord('g'), ord('G')):
                        print('here < ')
                        self.images_pd.at[record_index, 'Classification'] = 'G'
                    if key in (ord('b'), ord('B')):
                        self.images_pd.at[record_index, 'Classification'] = 'B'
                    self.images_pd = file_ops(self.images_pd, record_index, excel_file)
                    change_classification = False
                    record_index = new_record_index
                    break
                elif change_classification and new_record_index == num_images:
                    print('cc =:', change_classification, new_record_index, num_images, key)
                    if key in (ord('g'), ord('G')):
                        print('here ==')
                        images_pd.at[record_index, 'Classification'] = 'G'
                    if key in (ord('b'), ord('B')):
                        images_pd.at[record_index, 'Classification'] = 'B'
                    self.images_pd = file_ops(images_pd, record_index, excel_file)
                    cv2.destroyAllWindows()
                    return

                elif new_record_index == num_images and not change_classification:
                    # Save the DataFrame back to the Excel file at the end
                    self.images_pd.to_excel(excel_file, index=False)
                    cv2.destroyAllWindows()
                    return

                else:
                    record_index = new_record_index
                    break

            # # If the flag is set, terminate the loop
            #     if should_terminate:
            #         images_pd.to_excel(excel_file, index=False)
            #         break

        # # Save the DataFrame back to the Excel file at the end
        # images_pd.to_excel(excel_file, index=False)        

