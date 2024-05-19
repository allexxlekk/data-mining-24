import pandas as pd
import os
import numpy as np
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow_decision_forests.keras import pd_dataframe_to_tf_dataset
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

# Redefine labels to be contiguous integers starting from 0
label_mapping_old = {
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

label_mapping = {
    1: "Walking",
    2: "Running",
    3: "Shuffling",
    4: "Stairs (ascending)",
    5: "Stairs (descending)",
    6: "Standing",
    7: "Sitting",
    8: "Lying",
    13: "Cycling (sit)",
    14: "Cycling (stand)",
    130: "Cycling (sit, inactive)",
    140: "Cycling (stand, inactive)",
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

        data.append(df.values)

    # Combine data from all CSV files into a single array
    combined_data = np.concatenate(data)
    comb_df = pd.DataFrame(combined_data, columns=df.columns)

    # Convert columns to specified data types
    # for column in comb_df.columns.to_list():
    #     if column == "timestamp":
    #         continue
    #     comb_df[column] = comb_df[column].astype(float)

    # print("Mapping label numbers to strings...")
    # comb_df["label"] = comb_df["label"].replace(label_mapping)
    # print("Finished mapping")

    if train_subjects is not None and test_subjects is not None:
        subject_split = True
        total_subjects = train_subjects + test_subjects
        unique_subject_ids = comb_df["ID"].unique()
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

    # One-hot encode the label and ID columns
    labels_OHE = pd.get_dummies(comb_df["label"], dtype=float)  # , prefix="label")
    labels_OHE.columns = labels_OHE.columns.map(label_mapping)
    id_OHE = pd.get_dummies(comb_df["ID"], dtype=float, prefix="id")
    label_columns = labels_OHE.columns.tolist()

    # Standardize numerical features
    numerical_columns = FEATURES.copy()
    nn_features = FEATURES.copy()
    if "ID" in FEATURES:
        numerical_columns.remove("ID")
        nn_features.remove("ID")
        id_columns = id_OHE.columns.tolist()
        nn_features += id_columns

    # Standardize numerical columns
    scaler = StandardScaler()
    comb_df[numerical_columns] = scaler.fit_transform(comb_df[numerical_columns])

    # Concatenate one-hot encoded columns with the original dataframe
    data = {}
    data["df"] = pd.concat([comb_df, labels_OHE, id_OHE], axis=1)

    # Plot sensor values for each subject
    # for id in data["df"]["ID"].unique():
    #     plotSensorValues(data["df"][data["df"]["ID"] == id].iloc[:2_000], id)

    # Split dataset by subjects (some subjects are used for training and some for testing)
    if subject_split:
        # Seperate the data into train and test subjects based on user's choice
        first_ids_df = data["df"][
            data["df"]["ID"].isin(unique_subject_ids[:train_subjects])
        ]
        next_ids_df = data["df"][
            data["df"]["ID"].isin(
                unique_subject_ids[train_subjects : train_subjects + test_subjects]
            )
        ]
        # Shuffle dataframe rows
        data["train_df"] = first_ids_df.sample(frac=1, random_state=7).reset_index(
            drop=True
        )
        data["test_df"] = next_ids_df.sample(frac=1, random_state=7).reset_index(
            drop=True
        )
    else:
        # Shuffle dataframe rows
        # data["df"] = data["df"].sample(frac=1).reset_index(drop=True)
        # Calculate the index to split the dataframe
        split_index = int(0.7 * len(data["df"]))
        # Split the dataframe into train and test sets
        data["train_df"] = data["df"].iloc[:split_index]
        data["test_df"] = data["df"].iloc[split_index:]

    data["nn_features"] = nn_features
    data["X_train"] = np.array(data["train_df"][FEATURES])
    data["X_test"] = np.array(data["test_df"][FEATURES])
    data["y_train"] = np.array(data["train_df"]["label"])
    data["y_test"] = np.array(data["test_df"]["label"])
    ds_features = nn_features + ["label"]
    data["train_ds"] = pd_dataframe_to_tf_dataset(
        data["train_df"][ds_features], label="label"
    )
    data["test_ds"] = pd_dataframe_to_tf_dataset(
        data["test_df"][ds_features], label="label"
    )
    data["X_train_RNN"], data["y_train_RNN"] = segment_time_series(
        np.array(data["train_df"][nn_features]),
        np.array(data["train_df"][label_columns]),
        window_size,
    )
    data["X_test_RNN"], data["y_test_RNN"] = segment_time_series(
        np.array(data["test_df"][nn_features]),
        np.array(data["test_df"][label_columns]),
        window_size,
    )

    print("Training Features Shape:", data["X_train"].shape)
    print("Testing Features Shape:", data["X_test"].shape)
    print("Training Labels Shape:", data["y_train"].shape)
    print("Testing Labels Shape:", data["y_test"].shape)
    print("Training RNN Features Shape:", data["X_train_RNN"].shape)
    print("Testing RNN Features Shape:", data["X_test_RNN"].shape)
    print("Training RNN Labels Shape:", data["y_train_RNN"].shape)
    print("Testing RNN Labels Shape:", data["y_test_RNN"].shape)

    return data


def segment_time_series(data, labels_OHE, window_size, shuffle=True):
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
    num_classes = labels_OHE.shape[1]

    # Calculate the number of segments
    num_segments = num_samples - window_size + 1

    # Initialize arrays to store segmented data and labels
    segmented_data = np.zeros((num_segments, window_size, num_features))
    segmented_labels = np.zeros((num_segments, num_classes))

    # Segment the data
    for i in range(num_segments):
        start_idx = i
        end_idx = i + window_size
        segmented_data[i] = data[start_idx:end_idx, :]

        # Compute the majority label in the window
        window_labels = labels_OHE[start_idx:end_idx, :]
        majority_label_idx = np.argmax(np.sum(window_labels, axis=0))

        # Create a one-hot encoded vector for the majority label
        majority_label = np.zeros(num_classes)
        majority_label[majority_label_idx] = 1
        segmented_labels[i] = majority_label

    if shuffle:
        # After segmenting, shuffle the segmented data and labels together
        # Create an array of indices to shuffle
        shuffle_indices = np.random.permutation(num_segments)

        # Shuffle segmented_data and segmented_labels based on shuffle_indices
        segmented_data = segmented_data[shuffle_indices]
        segmented_labels = segmented_labels[shuffle_indices]

    return segmented_data, segmented_labels
