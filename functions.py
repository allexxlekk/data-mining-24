import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

COL_DTYPES = {
    "ID": "int",
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


def readDataset(folder_path="harth/") -> pd.DataFrame:
    files = os.listdir(folder_path)
    csv_files = [file for file in files if file.endswith(".csv")]
    data = []

    # Iterate over each CSV file and read it into a DataFrame
    for idx, csv_file in enumerate(csv_files):
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        # Add subject IDs
        df["ID"] = idx
        data.append(df[COL_DTYPES.keys()].values)

    # Combine data from all CSV files into a single array
    combined_data = np.concatenate(data)
    comb_df = pd.DataFrame(combined_data, columns=COL_DTYPES.keys())

    # Convert columns to specified data types
    for col, dtype in COL_DTYPES.items():
        comb_df[col] = comb_df[col].astype(dtype)

    return comb_df


def showDfMetrics(df: pd.DataFrame) -> None:
    print(df)
    df.info()
    print(df.describe())
    correlation_matrix = df.select_dtypes(include=["number"]).corr()

    # Plot the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()
