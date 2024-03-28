import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

COL_DTYPES = {
    "ID": "int",
    "time_step": "int",
    "timestamp": "object",
    "back_x": "float",
    "back_y": "float",
    "back_z": "float",
    "thigh_x": "float",
    "thigh_y": "float",
    "thigh_z": "float",
    "label": "int",
}

LABEL_DICT = {
    1: "walking",
    2: "running",
    3: "shuffling",
    4: "stairs (ascending)",
    5: "stairs (descending)",
    6: "standing",
    7: "sitting",
    8: "lying",
    13: "cycling (sit)",
    14: "cycling (stand)",
    130: "cycling (sit, inactive)",
    140: "cycling (stand, inactive)",
}


def convertTimestampToTimeStep(df: pd.DataFrame) -> pd.DataFrame:
    """Adds time_step column, indicating the amount of 0.01 second steps that have passed since the sensors started recording"""

    # Convert 'timestamp' column to datetime dtype
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Find the minimum timestamp value
    min_timestamp = df["timestamp"].min()

    # Calculate the time difference from the minimum timestamp and convert it to seconds
    df["time_step"] = (df["timestamp"] - min_timestamp).dt.total_seconds()

    # Convert time steps to increments of 0.01 seconds
    df["time_step"] = df["time_step"] / 0.01

    # Convert time steps to integers
    df["time_step"] = df["time_step"].astype(int)

    return df


def readDataset(folder_path="harth/", max_subjects=23) -> pd.DataFrame:
    """Reads original csv files and loads them into a dataframe."""

    print("Reading dataset...")
    files = os.listdir(folder_path)
    csv_files = [file for file in files if file.endswith(".csv")]
    data = []

    # Iterate over each CSV file and read it into a DataFrame
    for idx, csv_file in enumerate(csv_files):
        # Choose number of test subjects to include in the DataFrame (for quicker training purposes)
        if idx >= max_subjects:
            break
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        # Add subject IDs
        df["ID"] = idx
        df = convertTimestampToTimeStep(df)
        data.append(df[COL_DTYPES.keys()].values)

    # Combine data from all CSV files into a single array
    combined_data = np.concatenate(data)
    comb_df = pd.DataFrame(combined_data, columns=COL_DTYPES.keys())

    # Convert columns to specified data types
    for col, dtype in COL_DTYPES.items():
        comb_df[col] = comb_df[col].astype(dtype)

    return comb_df


def showDfMetrics(df: pd.DataFrame) -> None:
    """Lek's job to make this function meaningful."""

    print(df.head())
    df.info()
    df.describe()
