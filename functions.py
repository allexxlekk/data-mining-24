import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np


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


def preprocessTimestamp(df: pd.DataFrame) -> pd.DataFrame:
    date_time = pd.to_datetime(df["timestamp"])

    # Drop unneeded columns
    columns_to_drop = ["timestamp", "index", "Unnamed: 0"]
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # df["year"] = date_time.dt.year
    df["month"] = date_time.dt.month
    df["day"] = date_time.dt.day

    # Encode cyclic features as sine and cosine waves
    df["hour_sin"] = np.sin(2 * np.pi * date_time.dt.hour / 23.0)
    df["hour_cos"] = np.cos(2 * np.pi * date_time.dt.hour / 23.0)
    df["minute_sin"] = np.sin(2 * np.pi * date_time.dt.minute / 59.0)
    df["minute_cos"] = np.cos(2 * np.pi * date_time.dt.minute / 59.0)
    df["day_of_week_sin"] = np.sin(2 * np.pi * date_time.dt.dayofweek / 6.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * date_time.dt.dayofweek / 6.0)
    return df


def readDataset(folder_path="harth/", max_subjects=23) -> pd.DataFrame:
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
        # df = convertTimestampToTimeStep(df)
        df = preprocessTimestamp(df)
        data.append(df.values)

    # Combine data from all CSV files into a single array
    combined_data = np.concatenate(data)
    comb_df = pd.DataFrame(combined_data, columns=df.columns)

    # Redefine labels to be contiguous integers starting from 0
    label_mapping = {
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

    # Convert columns to specified data types
    for column in comb_df.columns.to_list():
        comb_df[column] = comb_df[column].astype(float)

    comb_df["label"] = comb_df["label"].replace(label_mapping)

    return comb_df


def showDfMetrics(df: pd.DataFrame) -> None:
    """Lek's job to make this function meaningful."""

    print(df.head())
    df.info()
    df.describe()
