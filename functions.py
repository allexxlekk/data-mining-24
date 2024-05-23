import pandas as pd
import os
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data_visualization import plotSensorValues

FEATURES = [
    # "ID",
    # "time_step",
    "back_x",
    "back_y",
    "back_z",
    "thigh_x",
    "thigh_y",
    "thigh_z",
]

# LABEL_MAPPING = {
#     1: "Walking",
#     2: "Running",
#     3: "Shuffling",
#     4: "Stairs (ascending)",
#     5: "Stairs (descending)",
#     6: "Standing",
#     7: "Sitting",
#     8: "Lying",
#     13: "Cycling (sit)",
#     14: "Cycling (stand)",
#     130: "Cycling (sit, inactive)",
#     140: "Cycling (stand, inactive)",
# }

LABEL_LIST = [
    "Walking",
    "Running",
    "Shuffling",
    "Stairs (ascending)",
    "Stairs (descending)",
    "Standing",
    "Sitting",
    "Lying",
    "Cycling (sit)",
    "Cycling (stand)",
    "Cycling (sit, inactive)",
    "Cycling (stand, inactive)",
]

LABEL_MAPPING = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    13: 8,
    14: 9,
    130: 10,
    140: 11,
}


def convertTimestampToTimeStep(df: pd.DataFrame) -> pd.DataFrame:
    """Adds time_step column, indicating the amount of 0.01 second steps that have passed since the sensors started recording"""

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Find the minimum timestamp value
    min_timestamp = df["timestamp"].min()

    # Calculate the time difference from the minimum timestamp and convert it to seconds
    df["time_step"] = (df["timestamp"] - min_timestamp).dt.total_seconds()

    # Convert time steps to increments of 0.01 seconds
    df["time_step"] = df["time_step"] / 0.01

    # Convert time steps to integers
    df["time_step"] = df["time_step"].astype(int)

    return df


def readAndPreprocessData(
    folder_path="harth/",
    max_subjects=23,
    train_subjects=None,
    test_subjects=None,
    window_size=50,
) -> dict:
    """Reads original csv files and loads them into a dataframe."""

    if max_subjects < 1:
        print(f"Invalid max_subjects: {max_subjects}")
        return -1

    print("Reading dataset...")
    files = os.listdir(folder_path)
    csv_files = [file for file in files if file.endswith(".csv")]
    data = []
    # Iterate over each CSV file and read it into a DataFrame
    for idx, csv_file in enumerate(csv_files):
        if idx >= max_subjects:
            break
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path, quoting=csv.QUOTE_NONE)
        # Add subject IDs
        df["ID"] = idx

        df = convertTimestampToTimeStep(df)
        # Drop unneeded columns
        columns_to_drop = ["index", "Unnamed: 0"]
        for col in columns_to_drop:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        if idx == 20:
            # Filter out odd timestamps for subject 20 to be consistent with 20ms sampling of the other subjects
            df = df[df["time_step"] % 2 == 0]

        data.append(df.values)

    # Combine data from all CSV files into a single array
    combined_data = np.concatenate(data)
    df = pd.DataFrame(combined_data, columns=df.columns)

    # Convert columns to specified data types
    df["label"] = df["label"].map(LABEL_MAPPING)
    df[FEATURES] = df[FEATURES].astype(float)
    int_cols = ["label", "ID", "time_step"]
    df[int_cols] = df[int_cols].astype(int)

    if train_subjects is not None and test_subjects is not None:
        subject_split = True
        total_subjects = train_subjects + test_subjects
        unique_subject_ids = df["ID"].unique()
        if total_subjects > len(unique_subject_ids):
            print(
                f"Train and test subjects exceed total subjects {len(unique_subject_ids)}!"
            )
            train_subjects = int(len(unique_subject_ids) * 2 / 3)
            test_subjects = int(len(unique_subject_ids) * 1 / 3)
            print(
                f"Setting train subjects to {train_subjects} and subjects to {test_subjects}"
            )
    else:
        subject_split = False

    ### Data Preprocessing
    # Concatenate one-hot encoded columns with the original dataframe
    data = {}

    # Plot sensor values for each subject
    # for id in new_df["ID"].unique():
    #     plotSensorValues(new_df[new_df["ID"] == id].iloc[:2_000], id)

    # Split dataset by subjects (some subjects are used for training and some for testing)
    if subject_split:
        # Seperate the data into train and test subjects based on user's choice
        train_df = df[df["ID"].isin(unique_subject_ids[:train_subjects])]
        test_df = df[
            df["ID"].isin(
                unique_subject_ids[train_subjects : train_subjects + test_subjects]
            )
        ]
        X_train, y_train = segment_time_series(
            np.array(train_df[FEATURES]),
            np.array(train_df["label"]),
            window_size,
        )
        X_test, y_test = segment_time_series(
            np.array(test_df[FEATURES]),
            np.array(test_df["label"]),
            window_size,
        )
        # X_train, y_train = segment_time_series_v2(train_df)
        # X_test, y_test = segment_time_series_v2(test_df)
    else:
        # X, y = segment_time_series_v2(new_df)
        X, y = segment_time_series(
            np.array(df[FEATURES]), np.array(df["label"]), window_size
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=7
        )

    print("Training Features Shape:", X_train.shape)
    print("Testing Features Shape:", X_test.shape)
    print("Training Labels Shape:", y_train.shape)
    print("Testing Labels Shape:", y_test.shape)

    return X_train, X_test, y_train, y_test


def segment_time_series(data, labels, window_size, shuffle=True):
    """
    Segment time series data into fully overlapping fixed-length sequences.

    Args:
        data (np.ndarray): 2D array of shape (num_samples, num_features) representing the accelerometer data for one time step.
        labels (np.ndarray): 2D array of shape (num_samples, num_classes) containing the one-hot encoded labels.
        window_size (int): Number of time steps to include in each segment.

    Returns:
        Tuple of segmented data and corresponding labels.
        Segmented data will be a 3D array of shape (num_segments, window_size, num_features).
        Segmented labels will be a 2D array of shape (num_segments, num_classes).
    """
    num_samples, num_features = data.shape
    # num_classes = labels.shape[1]

    # Calculate the number of segments
    num_segments = num_samples - window_size + 1

    # Initialize arrays to store segmented data and labels
    segmented_data = np.zeros((num_segments, window_size, num_features))
    segmented_labels = np.zeros((num_segments), dtype=int)

    # Segment the data
    for i in range(num_segments):
        start_idx = i
        end_idx = i + window_size
        segmented_data[i] = data[start_idx:end_idx, :]

        # Compute the majority label in the window
        window_labels = labels[start_idx:end_idx]
        # Use numpy.unique to find unique elements and their counts
        unique_values, counts = np.unique(window_labels, return_counts=True)

        # Find the index of the maximum count
        majority_label_idx = unique_values[np.argmax(counts)]

        # Create a one-hot encoded vector for the majority label
        # majority_label = np.zeros(num_classes)
        # majority_label[majority_label_idx] = 1
        segmented_labels[i] = majority_label_idx

    if shuffle:
        # After segmenting, shuffle the segmented data and labels together
        # Create an array of indices to shuffle
        shuffle_indices = np.random.permutation(num_segments)

        # Shuffle segmented_data and segmented_labels based on shuffle_indices
        segmented_data = segmented_data[shuffle_indices]
        segmented_labels = segmented_labels[shuffle_indices]

    return segmented_data, segmented_labels
