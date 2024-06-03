import pandas as pd
import os
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from data_visualization import plot_sensor_values
from time import time
import pickle

PP_DATA_FP = "preprocessed_timeseries/"

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
    subjects=22,
    window_length_ms=400,
    overlap=0.99,
) -> list:
    """Reads original csv files and loads them into a dataframe. Appropriate pre-processing is applied to the dataset and finally returns the inputs and labels of classifiers."""

    print("Reading dataset...")
    files = os.listdir(folder_path)
    csv_files = [file for file in files if file.endswith(".csv")]
    X = []
    y = []
    # Iterate over each CSV file and read it into a DataFrame
    for idx, csv_file in enumerate(csv_files):
        if idx >= subjects:
            break
        print(f"\nPreprocessing subject {idx}...")
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path, quoting=csv.QUOTE_NONE)
        # Add subject IDs
        df["ID"] = idx
        df["label"] = df["label"].map(LABEL_MAPPING)
        print(df["label"].unique())

        df = add_timesteps(df)

        if idx == 20:
            # Filter out odd timestamps for subject 20 to be consistent with 20ms sampling of the other subjects
            df = df[df["time_step"] % 2 == 0]

        X_temp, y_temp = segment_time_series(df, window_length_ms, overlap)
        X.append(X_temp)
        y.append(y_temp)

        with open(PP_DATA_FP + f"subject_{idx}.pkl", "wb") as f:
            pickle.dump((X_temp, y_temp), f)

    # Plot sensor values for each subject
    # for id in df["ID"].unique()[:2]:
    #     plot_sensor_values(df, id, plot_start=0, plot_end=2_000)

    return X, y


def segment_time_series_old(data, labels, window_size):
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


def segment_time_series(df, window_length_ms, overlap):
    """This functions creates windows of a given length and overlap fit for time-series classification tasks."""

    # Time the function
    start = time()

    # Calculate samples_per_window and step_size
    samples_per_window = window_length_ms // 20  # 20 ms intervals
    step_size = int(np.ceil(samples_per_window * (1 - overlap)))

    # Calculate the number of windows without overlap
    num_windows = len(df) // step_size - samples_per_window

    # Generate array indices for each window
    window_indices = np.arange(num_windows) * step_size
    window_end_indices = window_indices + samples_per_window

    time_steps = df["time_step"].values
    labels = df["label"].values
    feature_data = df[FEATURES].values

    # Extract segments and labels for each window
    segments = [
        feature_data[i:end] for i, end in zip(window_indices, window_end_indices)
    ]
    segment_labels = [
        pd.Series(labels[i:end]).mode()[0]
        for i, end in zip(window_indices, window_end_indices)
    ]

    # Check for windows with maximum time step difference > 3
    valid_windows = [
        np.amax(np.diff(time_steps[i:end])) <= 3
        for i, end in zip(window_indices, window_end_indices)
    ]

    # Filter out invalid windows
    segments = np.array(segments)[valid_windows]
    segment_labels = np.array(segment_labels)[valid_windows]

    end = time()

    print(f"Segmenting took {round(end-start)} seconds")
    print(f"Original samples: {len(df)}\n")
    print(f"Step size: {step_size}")
    print(f"Number of windows: {num_windows}")
    print(f"Windows dropped: {num_windows - np.sum(valid_windows)}")
    print(f"Valid windows: {segments.shape[0]}")

    return shuffle(segments, segment_labels, random_state=7)


def get_label_distribution(y: np.array):
    """Calculates and returns the label distribution of the provided array."""

    # Get unique label occurences
    _, y_counts = np.unique(y, return_counts=True)
    total_count = np.sum(y_counts)
    return [100 * count / total_count for count in y_counts]


def load_preprocessed_data() -> list[np.array]:
    X = []
    y = []
    for file in os.listdir(PP_DATA_FP):
        with open(PP_DATA_FP + file, "rb") as f:
            X_temp, y_temp = pickle.load(f)
        X.append(X_temp)
        y.append(y_temp)

    return X, y


def flatten_selected_slices(ar: np.array, start, end):
    """Selects a slice from the given multi-dimensional array along the first axis, and flattens it into a lower-dimensional array."""
    if ar.ndim < 2:
        raise ValueError("The input array must have at least 2 dimensions.")

    # Select the slice
    selected_slice = ar[start:end]

    # Calculate the new shape for flattening
    new_shape = (-1,) + ar.shape[2:]

    # Return the flattened slice
    return selected_slice.reshape(new_shape)


def split_data(
    X: np.array, y: np.array, train_s=None, test_s=None, val_s=None
) -> list[np.array]:

    # Seperate the data into train test and validation subjects
    if train_s is not None and test_s is not None and val_s is not None:
        total_subjects = train_s + test_s + val_s
        X_subjects = X.shape[0]
        if total_subjects > X_subjects:
            print(f"Train and test subjects exceed total subjects {X_subjects}!")
            train_s = int(X_subjects * 0.6)
            test_s = int(X_subjects * 0.3)
            val_s = int(X_subjects * 0.1)
            if val_s == 0:
                val_s = 1
            total_subjects = train_s + test_s + val_s

            while total_subjects < X_subjects:
                train_s += 1
            print(f"Setting train subjects to {train_s} and subjects to {test_s}")

    # print("Training Features Shape:", X_train.shape)
    # print("Testing Features Shape:", X_test.shape)
    # print("Training Labels Shape:", y_train.shape)
    # print("Testing Labels Shape:", y_test.shape)

    return X_train, X_test, y_train, y_test
