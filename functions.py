import pandas as pd
import os
import numpy as np
import csv
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
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


def read_and_preprocess_data(
    folder_path="harth/",
    subjects=22,
    window_length_ms=400,
    overlap=0.99,
) -> list:
    """Reads original csv files and loads them into a dataframe. Appropriate pre-processing is applied to the dataset. Finally, 2 lists containing the inputs and the labels of classifiers are returned."""

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

        # For subject 20, keep every other row to be consistent with 20ms sampling of the other subjects
        if idx == 20:
            df = df.iloc[::2]

        # Convert timestamp column to datetime datatype
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["label"] = df["label"].map(LABEL_MAPPING)


        # Apply preprocessing on data and prepare input and labels
        X_temp, y_temp = segment_time_series(df, window_length_ms, overlap)

        # Append current subject's input and labels to the total list
        X.append(X_temp)
        y.append(y_temp)

        with open(PP_DATA_FP + f"subject_{idx}.pkl", "wb") as f:
            pickle.dump((X_temp, y_temp), f)

    return X, y


def segment_time_series(df, window_length_ms, overlap):
    """This function creates windows of a given length and overlap fit for time-series data."""

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

    # Extract necessary values
    timestamps = df["timestamp"].values
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

    # Drop windows with maximum time difference > 20ms
    valid_windows = [
        np.amax(np.diff(timestamps[i:end]).astype("timedelta64[ms]").astype(int)) <= 20
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

def create_feature_matrix(y: np.array):
    """Creates feature matrix to use for clustering."""
    
    num_subjects = len(y)
    # Initialize the feature matrix
    feature_matrix = np.zeros((num_subjects, 12))
    # Fill the feature matrix
    for i, subject_data in enumerate(y):
        labels, counts = np.unique(subject_data, return_counts=True)
        feature_matrix[i, labels] = counts

    
    # Normalize the feature vectors
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    return feature_matrix