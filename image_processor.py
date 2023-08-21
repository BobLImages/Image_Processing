import pandas as pd
from file_functions_2 import *
from tkinter import messagebox


class ImageProcessor:

    # Column mapping dictionary (key: original column name, value: SQL-compliant column name)
    column_mapping = {
        'Original_Image': 'Original Image',
        'Grayscale_Image': 'Grayscale Image',
        'Denoised_Image': 'Denoised Image',
        'Dehazed_Image': 'Dehazed Image',
        'Sharpened_Image': 'Sharpened Image',
        # Add other column mappings here
        'Image_ID': 'Image ID',
        'File_Name': 'File Name',
        'Haze_Factor': 'Haze Factor',
        'Hough_Info':'Hough Info',
        'Harris_Corners':'Harris Corners',
        'Hough_Circles':'Hough Circles',
        'Contour_Info' :'Contour Info',        
        'F_Stop': 'F-stop',
        'Black_Pixels': 'Black Pixels',
        'Mid_tone_Pixels': 'Mid-tone Pixels',
        'White_Pixels':  'White Pixels',
        'Classification': 'Classification',
        'SHV': 'SHV',
        'Faces': 'Faces',
        'Bodies': "Bodies",
        'Eyes': 'Eyes',
        'ISO': 'ISO',
        'Exposure': 'Exposure',
        'Variance': 'Variance',
        'Orientation': 'Orientation',
        'Brightness':'Brightness',
        'Contrast': 'Contrast'

       
    }


    # Define the column names as class attributes
    column_names = [
        'Image_ID', 'File_Name', 'Orientation', 'Brightness', 'Contrast', 'Haze_Factor', 'Hough_Info',
        'Hough_Circles', 'Harris_Corners', 'Contour_Info', 'Laplacian', 'SHV', 'Variance',
        'Exposure', 'F_Stop', 'ISO', 'Black_Pixels', 'Mid_tone_Pixels', 'White_Pixels',
        'Faces', 'Eyes', 'Bodies', 'Focal_Length', 'Classification', 'Original_Image',
        'Grayscale_Image', 'Denoised_Image', 'Dehazed_Image', 'Sharpened_Image'
    ]

    # original_directory =

    # excel_file =

    # sql_file =

    # excel_exists =

    # sql_exists =











    def __init__(self):
        # Initialize the DataFrame with the predefined column names
        self.images_pd = pd.DataFrame(columns=self.column_names)
        # self.

    def load_file_for_display(self, index):
        """
        Load a file (image or other) and prepare it for display based on a specified index.

        Args:
            index (int): The index of the record in the DataFrame to process.

        Returns:
            np.ndarray: The loaded and resized file data.
        """
        
        print(index)
        # Get the file path and image orientation for the specified index
        file_path = self.images_pd.at[index, 'File_Name']
        image_orientation = self.images_pd.at[index, 'Orientation']
        image_id = self.images_pd.at[index, 'Image_ID']
        print(file_path, image_orientation, image_id)
       



        # Load the file
        image = cv2.imread(file_path)

        # Resize the image using the resize_image function
        if image_orientation == 'Landscape':
            resized_image, p = resize_image(image, scale_percent=30)  # Adjust the scale percent as desired
        elif image_orientation == 'Portrait':
            resized_image, p = resize_image(image, scale_percent=20)  # Adjust the scale percent as desired

        return resized_image

    def create_plots(self, excel_file, attributes):
        images_pd = self.load_data(excel_file)
        num_plots = len(attributes)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

        for idx, attribute in enumerate(attributes):
            ax = axes[idx]
            ax.scatter(self.images_pd['Image_ID'], self.images_pd[attribute])
            ax.set_xlabel('Image ID')
            ax.set_ylabel(attribute)
            ax.set_title(attribute)

        plt.tight_layout()
        plt.show()







    def generate_statistics_image(self, row, resized_image, background_color=(0, 0, 0), text_color=(255, 255, 255),
                                  text_size=0.5, text_thickness=1, text_position=(50, 50)):


        data = {'Attribute': [], 'Value': []}

        # Add the statistics to the DataFrame
        data['Attribute'].append('Image Statistics for directory:')
        data['Value'].append(os.path.dirname(row['File_Name']))
        data['Attribute'].append('')
        data['Value'].append('')
        for column, value in row.iloc[:-1].items():
            # Modify column names using column_mapping
            if column == 'File_Name':
                data['Attribute'].append(self.column_mapping['File_Name'] + ': ')
                data['Value'].append(os.path.basename(value))
            elif column in self.column_mapping:
                data['Attribute'].append(self.column_mapping[column] + ':')
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
                       overlay_start[1] + len(text_lines) * attribute_offset + 30)

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


    def get_sharpened_image(sharpened_image_data):

        # Convert the binary data to a NumPy array
        sharpened_image_array = np.frombuffer(sharpened_image_data, dtype=np.uint8)
        return cv2.imdecode(sharpened_image_array, flags=cv2.IMREAD_COLOR)


    def get_dataframe_records(self):
        return self.images_pd.values.tolist()    

    def load_data(self, excel_file=None, sql_file=None, table_name=None):
        if sql_file and table_name:
            # Load data from SQL
            conn = sqlite3.connect(sql_file)
            query = f"SELECT * FROM {table_name}"
            self.images_pd = pd.read_sql_query(query, conn)
            conn.close()
        elif excel_file:
            # Load data from Excel
            self.images_pd = pd.read_excel(excel_file)
        else:
            # Handle case where neither SQL nor Excel files are provided
            raise ValueError("Either Excel file or SQL file and table name must be provided.")
        
        return self.images_pd


    def save_dataframe(self, file_type, table_name=None):
        overwrite = False
        save_data = False

        if file_type == 'excel':
            file_ext = 'xlsx'
            folder_path = "D:/Image Data Files/"
            file_exists_msg = "The Excel file already exists. Do you want to overwrite it?"
            file_not_exists_msg = "The Excel file does not exist. Do you want to save the data?"
        elif file_type == 'sql':
            file_ext = 'db'
            folder_path = "D:/Image Data Files SQL/"
            file_exists_msg = "The SQL file already exists. Do you want to overwrite it?"
            file_not_exists_msg = "The SQL file does not exist. Do you want to save the data?"

        file_path = os.path.join(folder_path, os.path.basename(os.path.dirname(self.images_pd['File_Name'][0])) + f'.{file_ext}')

        print(file_path)

        if os.path.exists(file_path):
            # Prompt the user with a message box to confirm overwriting
            overwrite = messagebox.askyesno("File Exists", file_exists_msg)
            if not overwrite:
                print("Operation canceled. Existing file will not be overwritten.")
                return
        else:
            # Prompt the user with a message box to confirm saving the data
            save_data = messagebox.askyesno("File Does Not Exist", file_not_exists_msg)
            if not save_data:
                print("Data not saved.")
                return

        if overwrite or save_data:
            if file_type == 'excel':
                # Save to Excel
                columns_to_include = self.images_pd.columns[:-4]
                data_subset = self.images_pd[columns_to_include]                
                data_subset.to_excel(file_path, index=False)
            
            elif file_type == 'sql':
                # Connect to the SQLite database
                sqliteConnection = sqlite3.connect(file_path)

                # Replace spaces with underscores in column names
                self.images_pd.columns = self.images_pd.columns.str.replace(' ', '_')

                # Save the DataFrame to the SQL database
                try:
                    self.images_pd.to_sql(table_name, sqliteConnection, if_exists='replace', index=False)
                    print(f"Data from DataFrame has been inserted in {table_name} table")
                except sqlite3.Error as error:
                    print("Failed to insert data into sqlite table", error)
                finally:
                    sqliteConnection.close()
                    print("The SQLite connection is closed")

            print(f"Data saved to {file_type} file: {file_path}")
            # view_btn['state'] = 'normal'
            # view_btn.pack()

        return self.images_pd

    def create_dataframe(self, images, dir_path):
        # Your image processing code here
        scale = 20
        enhanced_actions = False
        batch_size = 25
        data_list = []

        # self.images_pd = pd.DataFrame(
        #     columns=['Image_ID', 'File_Name', 'Orientation', 'Brightness', 'Contrast', 'Haze_Factor', 'Hough_Info',
        #              'Hough_Circles', 'Harris_Corners', 'Contour_Info', 'Laplacian', 'SHV', 'Variance',
        #              'Exposure', 'F_Stop', 'ISO', 'Black_Pixels', 'Mid_tone_Pixels', 'White_Pixels',
        #              'Faces', 'Eyes', 'Bodies', 'Focal_Length', 'Classification', 'Original_Image',
        #              'Grayscale_Image', 'Denoised_Image', 'Dehazed_Image', 'Sharpened_Image'
        #             ])

            # Populate the DataFrame with the data
        for image in images:
            print(image.image_id)
            print(image.fname)
            data_list.append({
                'Image_ID': image.image_id +1,
                'File_Name': image.fname,
                'Orientation': image.orientation,
                'Brightness': round(image.brightness),
                'Contrast': round(image.contrast),
                'Haze_Factor': round(image.haze_factor, 2),
                'Hough_Info': image.hough_info[0],
                'Hough_Circles': image.hough_circles,
                'Harris_Corners': image.harris_corners,
                'Contour_Info': image.contour_info[0],
                'Laplacian': round(image.laplacian),
                'SHV': round(image.shv, 2),
                'Variance': image.variance,
                'Exposure': image.exposure,
                'F_Stop': image.fstop,
                'ISO': image.iso,
                'Black_Pixels': image.b_w[0],
                'Mid_tone_Pixels': image.b_w[1],
                'White_Pixels': image.b_w[2],
                'Faces': image.faces,
                'Eyes': image.eyes,
                'Bodies': image.bodies,
                'Focal_Length': image.focal_length,
                'Classification': image.classification,
                'Original_Image': image.convertToBinaryData()[0],
                'Grayscale_Image': image.convertToBinaryData()[1],
                'Denoised_Image': image.convertToBinaryData()[2],
                'Dehazed_Image': image.convertToBinaryData()[3],
                'Sharpened_Image': image.convertToBinaryData()[4],
            })


       # Create the DataFrame from the list of dictionaries
        self.images_pd = pd.DataFrame(data_list, columns = self.column_names)
        print("All Image_ID values in the Create_DataFrame:")
        print(self.images_pd['Image_ID'].tolist())

        # Print all the File_Name values in the DataFrame
        print("All File_Name values in the Create_DataFrame:")
        print(self.images_pd['File_Name'].tolist())

    










    def process_images(self, dir_path):
        # Your image processing code here
        scale = 20
        enhanced_actions = False
        batch_size = 25

        fnames = get_files(dir_path)
        images = load_images_batch(fnames, scale, enhanced_actions, batch_size)
        self.create_dataframe(images,dir_path)
        self.save_dataframe('excel')



    def view_images(self, dir_path, excel_file, key_handler, file_ops):
        images_pd = self.load_data(excel_file)

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
            # print('Outer Loop:', record_index)
            # print(row['File_Name'], row['Orientation'])
            print(row)
            resized_image = self.load_file_for_display(record_index)

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

            print(row,resized_image.shape)
            resized_image = self.generate_statistics_image(row, resized_image, background_color=(0, 0, 0),
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

