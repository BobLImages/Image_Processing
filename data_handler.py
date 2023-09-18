import pandas as pd
import os
from tkinter import messagebox
import sqlite3

class DataHandler:
    def __init__(self):
        self.data = None

    def check_xlsx_file(full_excel_file):
        return os.path.exists(full_excel_file)

    def load_image_for_display(image_path, image_orientation):
        image = cv2.imread(image_path)

        # Resize the image using the resize_image function
        if image_orientation == 'Landscape':
            resized_image, p = resize_image(image, scale_percent=30)  # Adjust the scale percent as desired
        if image_orientation == 'Portrait':
            resized_image, p = resize_image(image, scale_percent=20)  # Adjust the scale percent as desired
        #print(resized_image.shape[:2])


    def load_from_excel(self, dir_path, excel_file_path, sheet):
        print('here')
        try:
            # Read data from Excel into a DataFrame
            images_pd = pd.read_excel(excel_file_path, sheet_name = sheet)
            print(f"Data loaded from Excel: {excel_file_path}")
            #print(images_pd.head())

            # Additional processing or attribute setting here if needed

            return images_pd  # Return the loaded DataFrame

        except Exception as ex:
            print(f"Error loading data from Excel: {str(ex)}")
            return None


        


    def load_from_sql(self, sql_file_path, table_name):
        
        print(sql_file_path, table_name)

        """
        Load data from an SQL file (SQLite database) and store it in the class instance's data attribute.
        Args:
            sql_file_path (str): Path to the SQL file (SQLite database file).
            table_name (str): Name of the SQL table to retrieve data from.

        Returns:
            None
        """
        try:
            # Establish a connection to the SQLite database
            conn = sqlite3.connect(sql_file_path)

            # Construct the SQL query to select data from the specified table
            query = f"SELECT * FROM {table_name}"

            # Read data from SQL into a DataFrame
            images_pd = pd.read_sql_query(query, conn)

            # Close the database connection
            conn.close()

            return images_pd  # Return the loaded DataFrame


        except Exception as e:
            print(f"Error loading data from SQL: {str(e)}")


    def save_data_to_excel(self, images_pd, excel_file_path,sheet):

        print("data_handler class save_data_to_excel method",'---------------',excel_file_path,'-----------------')
        print(images_pd)
        file_exists_msg = "The Excel file already exists. Do you want to overwrite it?"
        file_not_exists_msg = "The Excel file does not exist. Do you want to save the data?"
        overwrite = False
        save_data = False

        if os.path.exists(excel_file_path):
            # Prompt the user with a message box to confirm overwriting
            overwrite = messagebox.askyesno("File Exists", file_exists_msg)
            if not overwrite:
                print("Operation canceled. Existing file will not be overwritten.")
                return
        else:
            # Prompt the user with a message box to confirm saving the data
            save_data = messagebox.askyesno("File Does Not Exist", file_not_exists_msg)
            print(save_data, excel_file_path)
            if not save_data:
                print("Data not saved.")
                return


        print(excel_file_path)
        if overwrite or save_data:
            try:
                # Save to Excel
                print("data_handler class save to excel method", images_pd)
                columns_to_include = images_pd.columns[:-4]
                data_subset = images_pd[columns_to_include]                
                data_subset.to_excel(excel_file_path, sheet_name = sheet, index=False)
            except Exception as e:
                print(f"Error saving data to Excel: {str(e)}")




    def save_data_to_sql(self, images_pd, sql_file_path,table):

        print("data_handler class save_data_to_sql method",'---------------',sql_file_path,'-----------------')
        print(images_pd)
        file_exists_msg = "The SQL file already exists. Do you want to overwrite it?"
        file_not_exists_msg = "The SQL file does not exist. Do you want to save the data?"
        overwrite = False
        save_data = False

        if os.path.exists(sql_file_path):
            # Prompt the user with a message box to confirm overwriting
            overwrite = messagebox.askyesno("File Exists", file_exists_msg)
            if not overwrite:
                print("Operation canceled. Existing file will not be overwritten.")
                return
        else:
            # Prompt the user with a message box to confirm saving the data
            save_data = messagebox.askyesno("File Does Not Exist", file_not_exists_msg)
            print(save_data, sql_file_path)
            if not save_data:
                print("Data not saved.")
                return


        print(sql_file_path)
        if overwrite or save_data:
            try:
                # Save to sql
                print("data_handler class save to sql method", images_pd)
                # columns_to_include = images_pd.columns[:-4]
                # data_subset = images_pd[columns_to_include]                
                
                # Connect to an SQLite database (or create a new one if it doesn't exist)
                conn = sqlite3.connect(sql_file_path)

                # Save the DataFrame to a table in the database
                images_pd.to_sql(name=table, con=conn, if_exists='replace', index=False)

                # Close the database connection
                conn.close()

            except Exception as e:
                print(f"Error saving data to SQL: {str(e)}")





    # def save_dataframe(self, file_type, table_name=None):
    #     overwrite = False
    #     save_data = False

    #     if file_type == 'excel':
    #         file_ext = 'xlsx'
    #         folder_path = "D:/Image Data Files/"
    #         file_exists_msg = "The Excel file already exists. Do you want to overwrite it?"
    #         file_not_exists_msg = "The Excel file does not exist. Do you want to save the data?"
    #     elif file_type == 'sql':
    #         file_ext = 'db'
    #         folder_path = "D:/Image Data Files SQL/"
    #         file_exists_msg = "The SQL file already exists. Do you want to overwrite it?"
    #         file_not_exists_msg = "The SQL file does not exist. Do you want to save the data?"

    #     file_path = os.path.join(folder_path, os.path.basename(os.path.dirname(self.images_pd['File_Name'][0])) + f'.{file_ext}')

    #     print(file_path)

    #     if os.path.exists(file_path):
    #         # Prompt the user with a message box to confirm overwriting
    #         overwrite = messagebox.askyesno("File Exists", file_exists_msg)
    #         if not overwrite:
    #             print("Operation canceled. Existing file will not be overwritten.")
    #             return

    #     if overwrite or save_data:
    #         if file_type == 'excel':
    #             # Save to Excel
    #             columns_to_include = self.images_pd.columns[:-4]
    #             data_subset = self.images_pd[columns_to_include]                
    #             data_subset.to_excel(file_path, index=False)
            
    #         elif file_type == 'sql':
    #             # Connect to the SQLite database
    #             sqliteConnection = sqlite3.connect(file_path)

    #             # Replace spaces with underscores in column names
    #             self.images_pd.columns = self.images_pd.columns.str.replace(' ', '_')

    #             # Save the DataFrame to the SQL database
    #             try:
    #                 self.images_pd.to_sql(table_name, sqliteConnection, if_exists='replace', index=False)
    #                 print(f"Data from DataFrame has been inserted in {table_name} table")
    #             except sqlite3.Error as error:
    #                 print("Failed to insert data into sqlite table", error)
    #             finally:
    #                 sqliteConnection.close()
    #                 print("The SQLite connection is closed")

    #         print(f"Data saved to {file_type} file: {file_path}")
    #         # view_btn['state'] = 'normal'
    #         # view_btn.pack()

    #     return self.images_pd
