import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    mean_squared_error,
)

import numpy as np

# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.callbacks import EarlyStopping

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


def preprocessData(df: pd.DataFrame) -> list:
    """Prepares inputs (features and labels) for trainning and testing classifiers."""

    # Map integer values to label names
    df["label"] = df["label"].map(LABEL_DICT)

    # One-hot encode the label column
    one_hot_encoded = pd.get_dummies(df["label"], dtype=float)

    # Concatenate one-hot encoded columns with the original dataframe
    df = pd.concat([df, one_hot_encoded], axis=1)

    XY_df = df.drop("timestamp", axis=1, inplace=False)

    # Select columns after the "label" column
    label_index = df.columns.get_loc("label")
    classes = df.columns[label_index + 1 :].tolist()

    X = np.array(XY_df[XY_df.columns[:-11]])
    y = np.array(XY_df[classes])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7
    )

    print("Training Features Shape:", X_train.shape)
    print("Testing Features Shape:", X_test.shape)
    print("Training Labels Shape:", y_train.shape)
    print("Testing Labels Shape:", y_test.shape)

    return [df, X_train, X_test, y_train, y_test, classes]


def evaluateClassifier(
    cl: RandomForestRegressor | GaussianNB, X_test, y_test, classes
) -> None:
    print("\nEvaluating classifier...")
    y_pred_proba = cl.predict(X_test)

    # Convert one-hot encoded y_test to 1D array of the correct labels
    y_correct = np.argmax(y_test, axis=1)

    # Convert probabilities to predicted classes
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_correct, y_pred)
    precision = precision_score(y_correct, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_correct, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_correct, y_pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    mse = mean_squared_error(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_correct, y_pred)
    class_rep = classification_report(y_correct, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"MSE: {mse:.4f}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_rep)
