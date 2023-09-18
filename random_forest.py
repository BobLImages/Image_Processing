#random_forest.py

import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report
from image_processor import ImageProcessor
from sklearn.metrics import confusion_matrix
import seaborn as sns



def generate_confusion_matrix(target, predictions):
    # Create a confusion matrix
    cm = confusion_matrix(target, predictions)
    print(f"True Negative (TN): {cm[0, 0]}")
    print(f"False Positive (FP): {cm[0, 1]}")
    print(f"False Negative (FN): {cm[1, 0]}")
    print(f"True Positive (TP): {cm[1, 1]}")

    # Display the confusion matrix using seaborn
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Generate the classification report
    print(classification_report(target, predictions))



def random_forest_train(images_pd):
    data = images_pd  # Get the DataFrame from the ImageProcessor instance
    # Load the training data from the provided Excel file or DataFrame
    if isinstance(data, pd.DataFrame):
        # Input is a DataFrame, so no need to read from the file
        df = data
    elif isinstance(data, str):
        # Input is an Excel file name, so read the Excel file into a DataFrame
        df = pd.read_excel(data, header=0)
    else:
        raise ValueError("Invalid input. Please provide a DataFrame or an Excel file name.")

    # Preprocess the data if necessary (e.g., encoding categorical features)
    # For this example, we'll assume the data is already preprocessed in the ImageProcessor class
    print("Column Names:", df.columns)  # Print the column names

    # Split the data into features and target
    target = df['Classification'].astype(str)
    columns_to_select = ['Orientation', 'Brightness', 'Contrast', 'Hough Info',
                         'Harris Corners', 'Contour Info', 'Laplacian', 'SHV', 'Variance', 'Exposure', 'F-Stop', 'ISO',
                         'Black Pixels', 'Mid-tone Pixels', 'White Pixels', 'Faces', 'Eyes', 'Bodies', 'Focal Length']
    features = df.loc[:, columns_to_select]
    features_encoded = pd.get_dummies(features, columns=['Orientation'])

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(features_encoded, target, test_size=0.20, random_state=42)

    # Train the random forest classifier on the training data
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # Evaluate the classifier's performance on the validation set (optional but useful for hyperparameter tuning)
    accuracy_train = rf.score(X_train, y_train)
    accuracy_valid = rf.score(X_valid, y_valid)
    print(f"Training Accuracy: {accuracy_train}")
    print(f"Validation Accuracy: {accuracy_valid}")

    # Return the trained random forest classifier
    print(X_train.columns)
    return rf, X_train.columns





def random_forest_test(images_pd,rf):
    
    data = images_pd  # Get the DataFrame from the ImageProcessor instance
    trained_classifier = rf

    # Load the test data from the provided Excel file or DataFrame
    if isinstance(data, pd.DataFrame):
        # Input is a DataFrame
        pass
    elif isinstance(data, str):
        # Input is an Excel file name, 
        # Read the Excel file into a DataFrame
        data = pd.read_excel(data, header=0)
        pass
    else:
        raise ValueError("Invalid input. Please provide a DataFrame or an Excel file name.")

    if 'Classification' in data.columns and 'U' in data['Classification'].unique():
        print("Unclassified images exist in the loaded data.")
    else:
        print("No unclassified images found in the loaded data.")
        
        # Remove rows with 'U' in the 'Classification' column (unclassified data)
        data_filtered = data[data['Classification'] != 'U']

        # Test data without 'U' in 'Classification'
        target = data_filtered['Classification'].astype(str)
        columns_to_select = ['Orientation', 'Brightness', 'Contrast', 'Hough Info',
                             'Harris Corners', 'Contour Info', 'Laplacian', 'SHV', 'Variance', 'Exposure', 'F-Stop', 'ISO',
                             'Black Pixels', 'Mid-tone Pixels', 'White Pixels', 'Faces', 'Eyes', 'Bodies', 'Focal Length']
        features = data_filtered.loc[:, columns_to_select]
        features_encoded = pd.get_dummies(features, columns=['Orientation'])

        # Use the trained classifier to make predictions on the test data
        predictions = trained_classifier.predict(features_encoded)

        # ... rest of the code to print or record the predictions ...
        # ... rest of the code to print or record the predictions ...

        accuracy = accuracy_score(target, predictions)
        print(f"Test Accuracy: {accuracy}")

        generate_confusion_matrix(target, predictions)



def random_forest_unclassified(images_pd, rf,training_columns):
    data = images_pd  # Get the DataFrame from the ImageProcessor instance
    trained_classifier = rf

    # Load the test data from the provided Excel file or DataFrame
    if isinstance(data, pd.DataFrame):
        # Input is a DataFrame
        pass
    elif isinstance(data, str):
        # Input is an Excel file name,
        # Read the Excel file into a DataFrame
        data = pd.read_excel(data, header=0)
        pass
    else:
        raise ValueError("Invalid input. Please provide a DataFrame or an Excel file name.")

    # Define a default value for target
    target = None

    if 'Classification' in data.columns and 'U' in data['Classification'].unique():
        print("Unclassified images exist in the loaded data.")
        # Remove rows with 'U' in the 'Classification' column (unclassified data)
        data_filtered = data[data['Classification'] == 'U']

        # Test data without 'U' in 'Classification'
        target = data_filtered['Classification'].astype(str)
        columns_to_select = ['Orientation', 'Brightness', 'Contrast', 'Hough Info',
                             'Harris Corners', 'Contour Info', 'Laplacian', 'SHV', 'Variance', 'Exposure', 'F-Stop', 'ISO',
                             'Black Pixels', 'Mid-tone Pixels', 'White Pixels', 'Faces', 'Eyes', 'Bodies', 'Focal Length']
        features = data_filtered.loc[:, columns_to_select]
        features_encoded = pd.get_dummies(features, columns=['Orientation'])

        # Ensure that the test data has the same columns as the training data
        missing_columns_train = set(training_columns) - set(features_encoded.columns)
        for column in missing_columns_train:
            features_encoded[column] = 0

        # Reorder the columns to match the order used during training
        features_encoded = features_encoded[training_columns]

        # Use the trained classifier to make predictions on the test data
        predictions = trained_classifier.predict(features_encoded)
        # ... rest of the code to print or record the predictions ...
    # Check if target is not None before calculating accuracy
        if target is not None:
            accuracy = accuracy_score(target, predictions)
            print(f"Test Accuracy: {accuracy}")

        # Create a confusion matrix
        generate_confusion_matrix(target, predictions)

    else:
        print("No unclassified images found in the loaded data.")


















#     if isinstance(data, pd.DataFrame):
#         # Input is a DataFrame
#         pass
#     elif isinstance(data, str):
#         # Input is an Excel file name, 
#         # Read the Excel file into a DataFrame
#         data = pd.read_excel(data, header = 0)
#         pass
#     else:
#         # Invalid input type
#         print("Invalid input. Please provide a DataFrame or an Excel file name.")

#     if 'Classification' in data.columns and 'U' in data['Classification'].unique():
#         print("Unclassified images exist in the loaded data.")
#     else:
#         print("No unclassified images found in the loaded data.")    




#     target = data['Classification'].astype(str)
#     columns_to_select = [ 'Orientation', 'Brightness', 'Contrast', 'Hough Info', 
#                          'Harris Corners', 'Contour Info', 'Laplacian', 'SHV', 'Variance', 'Exposure', 'F-Stop', 'ISO',
#                          'Black Pixels', 'Mid-tone Pixels', 'White Pixels', 'Faces', 'Eyes', 'Bodies', 'Focal Length']

#     features = data.loc[:, columns_to_select]
#     target.value_counts().plot(kind='bar')
#     plt.title('Target Variable Distribution')
#     plt.show()

#     features_encoded = pd.get_dummies(features, columns=['Orientation'])
#     X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.20, random_state=42)

#     rf = RandomForestClassifier()
#     rf.fit(X_train, y_train)

#     predictions = rf.predict(X_test)

#     accuracy = accuracy_score(y_test, predictions)
#     print(f"Accuracy: {accuracy}")


#     # Import necessary libraries for the confusion matrix
#     import seaborn as sns
#     from sklearn.metrics import confusion_matrix

#     # Create a confusion matrix
#     cm = confusion_matrix(y_test, predictions)
#     print(f"True Negative (TN): {cm[0, 0]}")
#     print(f"False Positive (FP): {cm[0, 1]}")
#     print(f"False Negative (FN): {cm[1, 0]}")
#     print(f"True Positive (TP): {cm[1, 1]}")
#     sns.heatmap(cm, annot=True, fmt='d')
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.show()
#     # Get the indices of false negatives
#     false_indices = [index for index, (actual, predicted) in enumerate(zip(y_test, predictions)) if actual != predicted]
#     print(false_indices)
#     print(len(false_indices))
#     print(predictions)
#     print(len(predictions))
#     print (y_test)

#     true_indices = [index for index, (actual, predicted) in enumerate(zip(y_test, predictions)) if actual == predicted]

#     fp = 0
#     fn = 0
#     fp_array = []
#     fn_array = []

#     tp = 0
#     tn = 0
#     tp_array = []
#     tn_array = []

#     fp_array = [t for t in false_indices if predictions[t] == 'G']
#     fn_array = [t for t in false_indices if predictions[t] == 'B']

#     fp = len(fp_array)
#     fn = len(fn_array)

#     print('False Positives',fp)
#     print('False Negatives',fn)


#     print(classification_report(y_test, predictions))

#     tp_array = [t for t in true_indices if predictions[t] == 'G']
#     tn_array = [t for t in true_indices if predictions[t] == 'B']

#     tp = len(tp_array)
#     tn = len(tn_array)

#     print('True Positives',tp)
#     print('True Negatives',tn)

#     print(fn_array)
#     print(fp_array)
#     print(tn_array)
#     print(tp_array)

# def print_results(label, index_array, data, predictions):
#     print(label)
#     for index, sample in enumerate(index_array):
#         fname = data.iloc[sample]['fname']
#         classification = data.iloc[sample]['classification']
#         test_results = predictions[sample]
#         print(f"Index: {sample}, fname: {fname}, classification: {classification}, test results: {test_results}")

#     # Call the function for each array of indices
#     print_results("False Positives:", fp_array, data, predictions)
#     print_results("False Negatives:", fn_array, data, predictions)
#     print_results("True Positives:", tp_array, data, predictions)
#     print_results("True Negatives:", tn_array, data, predictions)
