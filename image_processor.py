import pandas as pd
from file_functions_2 import *
from tkinter import messagebox
from data_handler import *

    


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
        'Contrast': 'Contrast',
        'Laplacian': 'Laplacian'

       
    }


    # Define the column names as class attributes
    column_names = [
        'Image_ID', 'File_Name', 'Orientation', 'Brightness', 'Contrast', 'Haze_Factor', 'Hough_Info',
        'Hough_Circles', 'Harris_Corners', 'Contour_Info', 'Laplacian', 'SHV', 'Variance',
        'Exposure', 'F_Stop', 'ISO', 'Black_Pixels', 'Mid_tone_Pixels', 'White_Pixels',
        'Faces', 'Eyes', 'Bodies', 'Focal_Length', 'Classification', 'Original_Image',
        'Grayscale_Image', 'Denoised_Image', 'Dehazed_Image', 'Sharpened_Image'
    ]


    statistics_window = None
    sort_order = None

    @classmethod
    def initialize(cls, dir_path):
        cls.original_directory = dir_path
        parent_directory, data_file_name = os.path.split(dir_path)
        cls.excel_path = "d:/Image Data Files/" + data_file_name + '.xlsx'
        cls.sql_path = "d:/Image Data Files sql/" + data_file_name + '.db'
        cls.table_sheet = data_file_name 
        cls.accepted_dir = os.path.join(dir_path, 'Accept')
        cls.rejected_dir = os.path.join(dir_path, 'Reject')

        # Check whether xlsx exists
        cls.excel_exists = os.path.exists(cls.excel_path)
        cls.sql_exists = os.path.exists(cls.sql_path)


    @classmethod
    def get_cumlative_stats(cls,images_pd):
        # Specify the columns you want to calculate statistics for
        selected_columns = [
            'Brightness', 'Contrast', 'Haze_Factor', 'Hough_Info',
            'Hough_Circles', 'Harris_Corners', 'Contour_Info',
            'Laplacian', 'SHV', 'Variance',
            'Black_Pixels', 'Mid_tone_Pixels', 'White_Pixels'
        ]

        # Calculate mean, min, max, and std for the selected columns
        cls.stats = images_pd[selected_columns].agg(['mean', 'min', 'max', 'std'])
        # Transpose the statistics DataFrame to have columns as rows
        cls.stats = cls.stats.transpose()

        # Rename the columns for clarity
        cls.stats.columns = ['Mean', 'Min', 'Max', 'Std']

        # print(cls.stats)


    def __init__(self):
        # Initialize the DataFrame with the predefined column names
        self.images_pd = pd.DataFrame(columns=self.column_names)
        self.sorted_pd = None

    def load_data_from_data_handler(self, source):

       # print('--',source)
        # Call the DataHandler to load data
        data_handler = DataHandler()
        if source == "excel":
            self.images_pd = data_handler.load_from_excel(ImageProcessor.original_directory, ImageProcessor.excel_path,ImageProcessor.table_sheet)
        elif source == "sql":
            self.images_pd = data_handler.load_from_sql(ImageProcessor.sql_path, ImageProcessor.table_sheet)
        else:
            # Handle an invalid source if needed
            self.images_pd = None

        # print(self.images_pd)
        ImageProcessor.get_cumlative_stats(self.images_pd)




        return self.images_pd



    def save_data_from_data_handler(self,source):

        # print(self.images_pd)
        # Call the DataHandler to load data
        data_handler = DataHandler()
        if source == "excel":
            data_handler.save_data_to_excel(self.images_pd, self.excel_path,self.table_sheet)
        elif source == "sql":
            data_handler.save_data_to_sql(self.images_pd, self.sql_path,self.table_sheet)
        else:
            # Handle an invalid source if needed
            self.images_pd = None

        return self.images_pd


    def load_file_for_display(self, index):
        """
        Load a file (image or other) and prepare it for display based on a specified index.

        Args:
            index (int): The index of the record in the DataFrame to process.

        Returns:
            np.ndarray: The loaded and resized file data.
        """
        
        # print(index)
        # Get the file path and image orientation for the specified index
        file_path = self.sorted_pd.at[index, 'File_Name']
        image_orientation = self.sorted_pd.at[index, 'Orientation']
        image_id = self.sorted_pd.at[index, 'Image_ID']
        # print(file_path, image_orientation, image_id)

        # Load the file
        image = cv2.imread(file_path)

        # Resize the image using the resize_image function
        if image_orientation == 'Landscape':
            resized_image, p = resize_image(image, scale_percent=30)  # Adjust the scale percent as desired
        elif image_orientation == 'Portrait':
            resized_image, p = resize_image(image, scale_percent=20)  # Adjust the scale percent as desired

        return resized_image



    def create_plots(self, attributes):
        num_plots = len(attributes)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

        # Define color mapping for classifications
        color_map = {'G': 'green', 'B': 'red', 'U': 'blue'}

        for idx, attribute in enumerate(attributes):
            ax = axes[idx]
            
            # Get the classifications for this attribute (assuming it's in a column named 'Classification')
            classifications = self.images_pd['Classification']
            
            # Map the classifications to colors using the color_map
            colors = classifications.map(color_map)
            
            ax.scatter(self.images_pd['Image_ID'], self.images_pd[attribute], c=colors)
            ax.set_xlabel('Image ID')
            ax.set_ylabel(attribute)
            ax.set_title(attribute)

        plt.tight_layout()
        plt.show()



    def generate_statistics_image(self, row, background_color=(0, 0, 0), text_color=(255, 255, 255),
                                  text_size=0.5, text_thickness=1, text_position=(50, 50)):

       # Check if a statistics window is open and close it if it exists
        if self.statistics_window is not None:
            cv2.destroyWindow(self.statistics_window)
            self.statistics_window = None

        data = {'Attribute': [], 'Value': []}

        # Add the statistics to the DataFrame
        data['Attribute'].append('')
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

        # Define fixed widths for columns
        attribute_width = 30
        value_width = 15

        # Convert the DataFrame to a formatted table with fixed widths
        table_text = ''
        for attribute, value in zip(data['Attribute'], data['Value']):
            attribute = attribute[:attribute_width].rjust(attribute_width)
            value = value[:value_width].ljust(value_width)
            table_text += attribute + value + '\n'

        # Remove the trailing newline
        table_text = table_text.rstrip()        


        # Convert the DataFrame to a formatted table
        # table_text = df.to_string(index=False, header=False, justify='left')
        # print(table_text)
        
        # Split the table text into lines
        text_lines = table_text.split('\n')

        # Calculate the dimensions of the statistics image
        attribute_offset = 20  # Offset for the attribute column
        value_offset = 200  # Offset for the value column

        # Calculate the maximum text width
        max_text_width = max(
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)[0][0]
            for line in text_lines
        )


        # Calculate the overlay coordinates on the resized image
        overlay_start = (text_position[0], text_position[1])
        overlay_end = (overlay_start[0] + value_offset + max_text_width - 5,
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
                    # print(attribute, '-', value)
                else:
                    attribute = line.strip()
                    value = ""  # Set a default value if needed

                # Strip the first 10 characters from the attribute and remove underscores
                attribute = attribute[10:].strip('_')
                # print(attribute)
                
                attribute_position = (0, (i + 1) * attribute_offset)
                value_position = (value_offset, (i + 1) * attribute_offset)
                # print(attribute_position,value_position)

                cv2.putText(statistics_image, value, value_position, cv2.FONT_HERSHEY_SIMPLEX,
                            text_size, text_color, text_thickness)
                cv2.putText(statistics_image, attribute, attribute_position, cv2.FONT_HERSHEY_SIMPLEX,
                            text_size, text_color, text_thickness)
        return statistics_image

    def get_sharpened_image(sharpened_image_data):

        # Convert the binary data to a NumPy array
        sharpened_image_array = np.frombuffer(sharpened_image_data, dtype=np.uint8)
        return cv2.imdecode(sharpened_image_array, flags=cv2.IMREAD_COLOR)


    def get_dataframe_records(self):
        return self.images_pd.values.tolist()    



    def create_dataframe(self, images, dir_path):
        # Your image processing code here
        scale = 20
        enhanced_actions = False
        batch_size = 25
        data_list = []

    # Populate the DataFrame with the data
        for image in images:
            # print(image.image_id)
            # print(image.fname)
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
        ImageProcessor.initialize(dir_path)

        
        # print(ImageProcessor.original_directory)

        # print("All Image_ID values in the Create_DataFrame:")
        # print(self.images_pd['Image_ID'].tolist())

        # # Print all the File_Name values in the DataFrame
        # print("All File_Name values in the Create_DataFrame:")
        # print(self.images_pd['File_Name'].tolist())

        return self.images_pd

    

    def process_images(self, dir_path,excel_path,sql_path):
        # Your image processing code here
        scale = 20
        enhanced_actions = False
        batch_size = 25

        fnames = get_files(dir_path)
        images = load_images_batch(fnames, scale, enhanced_actions, batch_size)
        self.images_pd = self.create_dataframe(images,dir_path)
        # print("Image Processor process_images method", self.images_pd)
        


        self.save_data_from_data_handler('excel')
        self.save_data_from_data_handler('sql')



    def view_images(self, data_source, sort_source,key_handler):
        print(data_source)
        self.images_pd = self.load_data_from_data_handler(data_source)
        if sort_source == 'original':
            self.sorted_pd = self.images_pd
        elif sort_source == 'brightness':
            self.sorted_pd = self.images_pd.sort_values(by='Brightness')
        elif sort_source == 'contours':
            self.sorted_pd = self.images_pd.sort_values(by='Contour_Info')
        elif sort_source == 'laplacian':
            self.sorted_pd = self.images_pd.sort_values(by='Laplacian')

        # Add a new column 'original_index' to store the original index values
        self.sorted_pd['original_index'] = self.sorted_pd.index

        # Reset the index
        self.sorted_pd.reset_index(drop=True, inplace=True)

        # Print column names
        print("Column names of Sorted DataFrame:")
        print(self.sorted_pd.columns)


        print("Sorted DataFrame with reset index:")
        print(self.sorted_pd)



        rows = 5
        columns = 7

        landscape_data = create_grids(1344, 2018, rows, columns)
        portrait_data = create_grids(1344, 896, columns, rows)

        os.makedirs(ImageProcessor.accepted_dir, exist_ok=True)
        os.makedirs(ImageProcessor.rejected_dir, exist_ok=True)

        record_index = 0
        num_images = len(self.sorted_pd)

        while record_index < num_images:
            # row = self.images_pd.iloc[record_index]
            row = self.sorted_pd.iloc[record_index]
            image_id = row['Image_ID']
            

            # print('Outer Loop:', record_index)
            # print(row['File_Name'], row['Orientation'])
            #print(row)
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

            # print(row,resized_image.shape)
            statistics_image = self.generate_statistics_image(row, background_color=(0, 0, 0),
                                                      text_color=(0, 255, 255),
                                                      text_size=0.5, text_thickness=1, text_position=(10, 10))

            # Show the resized and statistics_images
            

            converted_image = cv2.convertScaleAbs(resized_image)


            # Get the width and height of the modified image
            height, width, _ = converted_image.shape

            cv2.imshow("Modified Image", converted_image)
            cv2.imshow('Image Statistics', statistics_image)
            
            # Set the position of the "Modified Image" window
            cv2.moveWindow("Modified Image", 0, 0)  # Place it at the upper-left corner (0, 0)

            # Set the position of the "Image Statistics" window
            # Assuming that the "Modified Image" window ends at (width, height)
            cv2.moveWindow('Image Statistics', width, 0)  # Place it at the top where the other window ends


            change_classification = False  # Initialize the flag for each image
            should_terminate = False

            while True:
                print('Waiting...')
                # Wait for a key event
                key = cv2.waitKeyEx(0)

                change_classification, key = key_handler(key)

                # print(key)
                if key == 2621440:
                    new_record_index = record_index + 1
                if key == 2490368:
                    new_record_index = record_index - 1
                elif key == 2293760:
                    new_record_index = num_images - 1
                elif key == 2359296:
                    new_record_index = 0
                elif key == 27:
                    self.images_pd.to_excel(ImageProcessor.excel_path, index=False)
                    self.sorted_pd = None
                    cv2.destroyAllWindows()
                    return 
                else:
                    new_record_index = record_index + 1

                # If change_classification is True, perform file operations and reset the flag
                if change_classification and new_record_index != num_images:
                    # print('cc =:', change_classification, new_record_index, num_images, key)
                    if key in (ord('g'), ord('G')):
                        self.sorted_pd.at[record_index, 'Classification'] = 'G'
                        original_index = self.sorted_pd.loc[record_index, 'original_index']
                        self.images_pd.at[original_index, 'Classification'] = 'G'

                    if key in (ord('b'), ord('B')):
                        self.sorted_pd.at[record_index, 'Classification'] = 'B'
                        original_index = self.sorted_pd.loc[record_index, 'original_index']
                        self.images_pd.at[original_index, 'Classification'] = 'B'

                    self.file_ops(original_index)
                    change_classification = False
                    record_index = new_record_index
                    break
                elif change_classification and new_record_index == num_images:
                    # print('cc =:', change_classification, new_record_index, num_images, key)
                    if key in (ord('g'), ord('G')):
                        self.sorted_pd.at[record_index, 'Classification'] = 'G'
                        original_index = self.sorted_pd.loc[record_index, 'original_index']
                        self.images_pd.at[original_index, 'Classification'] = 'G'

                    if key in (ord('b'), ord('B')):
                        self.sorted_pd.at[record_index, 'Classification'] = 'B'
                        original_index = self.sorted_pd.loc[record_index, 'original_index']
                        self.images_pd.at[original_index, 'Classification'] = 'B'

                    self.file_ops(original_index)
                    self.sorted_pd = None
                    cv2.destroyAllWindows()
                    return

                elif new_record_index == num_images and not change_classification:
                    record_index = 0
                    break




                    # # Save the DataFrame back to the Excel file at the end
                    # self.images_pd.to_excel(ImageProcessor.excel_path, index=False)
                    # cv2.destroyAllWindows()
                    # return

                else:
                    record_index = new_record_index
                    break

            # # If the flag is set, terminate the loop
            #     if should_terminate:
            #         images_pd.to_excel(excel_file, index=False)
            #         break

        # # Save the DataFrame back to the Excel file at the end
        # images_pd.to_excel(excel_file, index=False)        



    def file_ops(self,index):
        original_file_path = self.images_pd.at[index, 'File_Name']
        file_name = os.path.basename(original_file_path)
        parent_directory = os.path.dirname(original_file_path)

        # Determine the classification prefix and the destination directory
        classification = self.images_pd.at[index, 'Classification']
        if classification == 'G':
            destination_dir = ImageProcessor.accepted_dir
        elif classification == 'B':
            destination_dir = ImageProcessor.rejected_dir
        else:
            destination_dir = ImageProcessor.original_directory
        
        # Create the destination directory if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)

        # Rename and copy the original file by adding the classification prefix
        new_file_path = os.path.join(destination_dir, classification + file_name)
        print("Renaming and copying:", original_file_path, "to", new_file_path)

        # Perform the actual copy from the original file to the new destination
        shutil.copy(original_file_path, new_file_path)

        # Save the updated DataFrame back to the Excel file
        self.images_pd.to_excel(ImageProcessor.excel_path, index=False)
     


   # def initialize(cls, dir_path):
   #      cls.original_directory = dir_path
   #      parent_directory, data_file_name = os.path.split(dir_path)
   #      cls.excel_path = "d:/Image Data Files/" + data_file_name + '.xlsx'
   #      cls.sql_path = "d:/Image Data Files sql/" + data_file_name + '.db'
   #      cls. = os.path.join(dir_path, 'Accept')
   #      cls.rejected_dir = os.path.join(dir_path, 'Reject')


    # Return the updated DataFrame with the changed classifications
    #return images_pd
