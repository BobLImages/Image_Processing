import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report
from file_functions_2 import *
from segment_class import *


data = pd.read_excel('D:/Image Data Files/2023-05-05 Glen Allen vs Deep Run Girls Soccer.xlsx', header = 0)

plt.style.use('seaborn')
plt.grid(True)  # Enable grid lines

g_lap_category = []
b_lap_category = []
g_cont_category = []
b_cont_category = []



# Iterate over the DataFrame rows
for _, row in data.iterrows():
    if row['classification'] == 'G':
        g_lap_category.append(round(row['Lap']))
        g_cont_category.append(row['fl'])
    else:
        b_lap_category.append(round(row['Lap']))
        b_cont_category.append(row['fl'])

# Create a scatter plot
plt.scatter(g_cont_category, g_lap_category, color='green', label='G')
plt.scatter(b_cont_category, b_lap_category, color='red', label='B')

# Set labels and title
plt.xlabel('Focal Length')
plt.ylabel('Laplacian')
plt.title('Laplacian & Focal Length')

# Set the x-axis tick locations and labels
tick_locations = range(70, 601, 25)
plt.xticks(tick_locations, tick_locations)


# Add legend
plt.legend()

# Display the plot
plt.show()



g_lap_category = []
b_lap_category = []
g_cont_category = []
b_cont_category = []

# Dictionary to store counts for red and green data points for each focal length
focal_length_counts = {}

# Iterate over the DataFrame rows
for _, row in data.iterrows():
    classification = row['classification']
    focal_length = row['fl']
    rounded_contours = round(row['Contours'])

    if classification == 'G':
        g_lap_category.append(rounded_contours)
        g_cont_category.append(focal_length)
    else:
        b_lap_category.append(rounded_contours)
        b_cont_category.append(focal_length)

    # Update the counts for focal length and classification
    if focal_length not in focal_length_counts:
        focal_length_counts[focal_length] = {'G': 0, 'B': 0}
    
    focal_length_counts[focal_length][classification] += 1

# Define the sorting criteria
sorting_criteria = 'green_percentage'  # Choose from 'green_count', 'total_count', 'green_percentage'

# Sort the focal_length_counts dictionary based on focal length in ascending order
sorted_counts_focal_length = sorted(focal_length_counts.items(), key=lambda x: x[0])

# Sort the focal_length_counts dictionary based on the selected criteria
if sorting_criteria == 'green_count':
    sorted_counts = sorted(sorted_counts_focal_length, key=lambda x: x[1]['G'], reverse=True)
elif sorting_criteria == 'total_count':
    sorted_counts = sorted(sorted_counts_focal_length, key=lambda x: sum(x[1].values()), reverse=True)
elif sorting_criteria == 'green_percentage':
    sorted_counts = sorted(sorted_counts_focal_length, key=lambda x: x[1]['G'] / sum(x[1].values()), reverse=True)
else:
    raise ValueError("Invalid sorting criteria.")

# Print the sorted results
for focal_length, counts in sorted_counts:
    green_count = counts['G']
    red_count = counts['B']
    total_count = green_count + red_count
    green_percentage = round(green_count / total_count * 100, 2)
    print(f"Focal Length: {focal_length}, Green Count: {green_count}, Red Count: {red_count}, Total: {total_count}, Green %: {green_percentage}")

# Create a scatter plot
plt.scatter(g_cont_category, g_lap_category, color='green', label='G')
plt.scatter(b_cont_category, b_lap_category, color='red', label='B')

aspect_ratio = 0.5  # You can adjust this value to set the desired aspect ratio
plt.gca().set_aspect(aspect_ratio)


# Set labels and title
plt.xlabel('Focal Length')
plt.ylabel('Contours')
plt.title('Contours & Focal Length')

# Set the x-axis tick locations and labels
tick_locations = range(70, 601, 25)
plt.xticks(tick_locations, tick_locations)

# Add legend
plt.legend()

# Display the plot
plt.show()


target =target = data.iloc[:, 23].astype(str)  # Select the first letter of the first column as the target variable

columns_to_select = [1,2,3,5,6,7,8,9,10,11,12,13,16,17,18,19,20,21,22]
features = data.iloc[:, columns_to_select]
# features['hough_ratio'] = features.iloc[:, 3] / features.iloc[:, 4]
# features = features.drop(features.columns[4], axis = 1)


target.value_counts().plot(kind='bar')
plt.title('Target Variable Distribution')
plt.show()




features_encoded = pd.get_dummies(features)

X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.20, random_state=42)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

predictions = rf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")


# Import necessary libraries for the confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Create a confusion matrix
cm = confusion_matrix(y_test, predictions)
print(f"True Negative (TN): {cm[0, 0]}")
print(f"False Positive (FP): {cm[0, 1]}")
print(f"False Negative (FN): {cm[1, 0]}")
print(f"True Positive (TP): {cm[1, 1]}")
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# # Get the indices of false negatives
# false_negatives_indices = [index for index, (actual, predicted) in enumerate(zip(y_test, predictions)) if actual != predicted and predicted.startswith('G')]

# # Assuming you have the indices of false negatives in a list called 'false_negatives_indices'
# first_5_false_negatives_indices = false_negatives_indices[:5]

# # Retrieve the corresponding samples
# false_negatives_samples = data.iloc[first_5_false_negatives_indices]

# # Visualize the samples
# for index, sample in false_negatives_samples.iterrows():
#     # Assuming you are working with image data
#     image_path = 'F:/Year 2023/2023-05-09 St Catherines vs Veritas Girls Soccer/' + sample['Fname']  # Replace 'image_path' with the appropriate column name for the image path

#     image = Image.open(image_path)
#     plt.imshow(image)
#     plt.title(f"False Negative #{index + 1}")
#     plt.show()



# Get the indices of false negatives
false_indices = [index for index, (actual, predicted) in enumerate(zip(y_test, predictions)) if actual != predicted]
print(false_indices)
print(len(false_indices))
print(predictions)
print(len(predictions))
print (y_test)

true_indices = [index for index, (actual, predicted) in enumerate(zip(y_test, predictions)) if actual == predicted]
# print(true_indices)
# print(len(true_indices))
# print(predictions)
# print(len(predictions))
# print (y_test)

fp = 0
fn = 0
fp_array = []
fn_array = []

tp = 0
tn = 0
tp_array = []
tn_array = []

fp_array = [t for t in false_indices if predictions[t] == 'G']
fn_array = [t for t in false_indices if predictions[t] == 'B']

fp = len(fp_array)
fn = len(fn_array)

print('False Positives',fp)
print('False Negatives',fn)


print(classification_report(y_test, predictions))





tp_array = [t for t in true_indices if predictions[t] == 'G']
tn_array = [t for t in true_indices if predictions[t] == 'B']

tp = len(tp_array)
tn = len(tn_array)

# print('True Positives',tp)
# print('True Negatives',tn)

# print(fn_array)
# print(fp_array)
# print(tn_array)
# print(tp_array)

print("False Positives:")
for index, sample in enumerate(fp_array):
    fname = data.iloc[sample]['fname']
    classification = data.iloc[sample]['classification']
    test_results = predictions[sample]
    print(f"Index: {sample}, fname: {fname}, classification: {classification}, test results: {test_results}")

# Print columns 'fname', 'classification', and 'test results' for each item in fn_array
print("False Negatives:")
for index, sample in enumerate(fn_array):
    fname = data.iloc[sample]['fname']
    classification = data.iloc[sample]['classification']
    test_results = predictions[sample]
    print(f"Index: {sample}, fname: {fname}, classification: {classification}, test results: {test_results}")

# Print columns 'fname', 'classification', and 'test results' for each item in tp_array
print("True Positives:")
for index, sample in enumerate(tp_array):
    fname = data.iloc[sample]['fname']
    classification = data.iloc[sample]['classification']
    test_results = predictions[sample]
    print(f"Index: {sample}, fname: {fname}, classification: {classification}, test results: {test_results}")

# Print columns 'fname', 'classification', and 'test results' for each item in tn_array
print("True Negatives:")
for index, sample in enumerate(tn_array):
    fname = data.iloc[sample]['fname']
    classification = data.iloc[sample]['classification']
    test_results = predictions[sample]
    print(f"Index: {sample}, fname: {fname}, classification: {classification}, test results: {test_results}")


'''The error you encountered indicates that the variable ax_enlarged is not defined at the point where you are trying to plot the modified image. 
To resolve this issue, make sure you define ax_enlarged before the loop starts. Here's an updated version of the code that 
includes the definition of ax_enlarged:
'''
# Loop over the samples
for sample in tn_array:
    # Get the image path
    image_path = 'F:/Year 2023/2023-05-05 Glen Allen vs Deep Run Girls Soccer/' + data.iloc[sample, 0]

    image = cv2.imread(image_path)

    # Resize the image using the resize_image function
    resized_image, p = resize_image(image, ratio=40)  # Adjust the scale percent as desired

    # Convert the image to a numpy array
    # image_array = np.array(resized_image)

    display_width = resized_image.shape[0]
    display_height = resized_image.shape[1]

    print(display_width,display_height)   

    # Define the number of rows and columns for the grid
    rows = 5
    columns = 7

    # Generate the northwest points of the grid using your existing function
    NW,zone_width,zone_height = create_grids(display_width, display_height, rows, columns)

    


    # Draw the rectangles on the resized image
    for nw_point in NW:
        # Calculate the coordinates of the top-left and bottom-right corners of the rectangle
        rectangle_start = nw_point
        rectangle_end = (nw_point[0] + zone_height, nw_point[1] + zone_width)
        cv2.rectangle(resized_image, rectangle_start, rectangle_end, (255, 255, 255), 2)

    for _, row in data.iterrows():
        if row['fname'] == os.path.basename(image_path):  # Match the image filename with the row in the DataFrame
            text = f"Image Statistics:\n"
            for column, value in row.items():
                text += f"{column}: {value}\n"

            # Create the image with statistics
            background_color = (0, 0, 0)  # Background color for the statistics image
            text_color = (255,255,255)  # Text Color            
            text_size = 1  # Adjust text size as needed
            text_thickness = 2  # Adjust text thickness as needed

            # Define the position of the text
            text_position = (50, 30)  # Adjust the position of the text as needed


            # Calculate the size of the text
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)

            # Adjust the coordinates of the rectangle
            rectangle_start = (text_position[0], text_position[1] - text_height - 11)
            rectangle_end = (text_position[0] + text_width + 20, text_position[1] -110)

            # # Draw the rectangle
            # cv2.rectangle(resized_image, rectangle_start, rectangle_end, (255, 255, 255), 2)

            # # Add the text
            # cv2.putText(resized_image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, text_thickness)

            text_lines = text.split('\n')
            text_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)[0][0] for line in text_lines)
            text_height = len(text_lines) * cv2.getTextSize(text_lines[0], cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)[0][1]

# statistics_image = np.zeros((text_height + 20, text_width + 20, 3), dtype=np.uint8)


#             text_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)[0] for line in text_lines)
#             text_height = len(text_lines) * cv2.getTextSize(text_lines[0], cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)[0][1]

            statistics_image = np.zeros((text_height + 200, text_width + 11, 3), dtype=np.uint8)
            statistics_image[:] = background_color

            text_position = (10, 10)
            text_color = (255, 255, 255)

            for i, line in enumerate(text_lines):
                cv2.putText(statistics_image, line, (text_position[0], text_position[1] + (i + 1) * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, text_thickness)

            # Overlay the statistics image onto the resized image
            overlay_start = text_position
            overlay_end = (overlay_start[0] + statistics_image.shape[1], overlay_start[1] + statistics_image.shape[0])
            resized_image[overlay_start[1]:overlay_end[1], overlay_start[0]:overlay_end[0]] = statistics_image

    # Show the modified image
    converted_image = cv2.convertScaleAbs(resized_image)
    cv2.imshow("Modified Image", converted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
