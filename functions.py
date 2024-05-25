import pandas as pd
import os
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from data_visualization import plot_sensor_values

FEATURES = [
    "back_x",
    "back_y",
    "back_z",
    "thigh_x",
    "thigh_y",
    "thigh_z",
]

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


def add_timesteps(df: pd.DataFrame) -> pd.DataFrame:
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


def read_and_preprocess_data(
    folder_path="harth/",
    max_subjects=23,
    train_subjects=None,
    test_subjects=None,
    window_size=50,
) -> list:
    """Reads original csv files and loads them into a dataframe. Appropriate pre-processing is applied to the dataset and finally returns the inputs and labels of classifiers."""

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

        df = add_timesteps(df)
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

    # Plot sensor values for each subject
    # for id in df["ID"].unique()[:2]:
    #     plot_sensor_values(df, id, plot_start=0, plot_end=2_000)

    data = {}
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
    else:
        X, y = segment_time_series(
            np.array(df[FEATURES]), np.array(df["label"]), window_size
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=7, shuffle=False
        )

    print("Training Features Shape:", X_train.shape)
    print("Testing Features Shape:", X_test.shape)
    print("Training Labels Shape:", y_train.shape)
    print("Testing Labels Shape:", y_test.shape)

    return X_train, X_test, y_train, y_test


def segment_time_series(data, labels, window_size):
    """Segment time series data into fully overlapping fixed-length sequences."""

    # Calculate the number of segments
    num_samples, num_features = data.shape
    num_segments = num_samples - window_size + 1

    # Initialize arrays to store segmented data and labels
    X_semented = np.zeros((num_segments, window_size, num_features))
    y_segmented = np.zeros((num_segments), dtype=int)

    # Segment the data
    for i in range(num_segments):
        start_idx = i
        end_idx = i + window_size
        X_semented[i] = data[start_idx:end_idx, :]

        # Compute the majority label in the window
        window_labels = labels[start_idx:end_idx]
        # Use numpy.unique to find unique elements and their counts
        unique_values, counts = np.unique(window_labels, return_counts=True)

        # Find the index of the maximum count
        majority_label_idx = unique_values[np.argmax(counts)]

        # Append the majority label to the segmented_labels list
        y_segmented[i] = majority_label_idx

    # Return the shuffled data and labels
    return shuffle(X_semented, y_segmented, random_state=7)


def get_label_distribution(y: np.array):
    """Calculates and returns the label distribution of the provided array."""

    # Get unique label occurences
    _, y_counts = np.unique(y, return_counts=True)
    total_count = np.sum(y_counts)
    return [100 * count / total_count for count in y_counts]
