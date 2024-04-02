import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
import tensorflow as tf
from keras.models import Sequential

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


def preprocessData_v2(
    df: pd.DataFrame, train_subjects=None, test_subjects=None
) -> list:
    """Prepares inputs (features and labels) for trainning and testing classifiers."""

    features = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]

    # One-hot encode the label column
    one_hot_encoded = pd.get_dummies(df["label"], dtype=float)

    # Concatenate one-hot encoded columns with the original dataframe
    df = pd.concat([df, one_hot_encoded], axis=1)

    # Get the "label" column index
    label_index = df.columns.get_loc("label")
    # Select columns after the "label" column
    classes = df.columns[label_index + 1 :].tolist()

    if train_subjects != None and test_subjects != None:
        unique_subject_ids = df["ID"].unique()
        if train_subjects + test_subjects > len(unique_subject_ids):
            print(
                f"Train and test subjects exceed total subjects (22)! Setting train subjects to 17 and test subjects to 5."
            )
            train_subjects = 17
            test_subjects = 5

        # Seperate the data into train and test subjects based on user's choice
        first_ids_df = df[df["ID"].isin(unique_subject_ids[:train_subjects])].drop(
            labels="ID", axis=1
        )
        next_ids_df = df[
            df["ID"].isin(
                unique_subject_ids[train_subjects : train_subjects + test_subjects]
            )
        ].drop(labels="ID", axis=1)
        # Shuffle dataframe rows
        train_df = first_ids_df.sample(frac=1).reset_index(drop=True)
        test_df = next_ids_df.sample(frac=1).reset_index(drop=True)
    else:
        # Shuffle dataframe rows
        df = df.sample(frac=1).reset_index(drop=True)
        # Calculate the index to split the dataframe
        split_index = int(0.7 * len(df))
        # Split the dataframe into train and test sets
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]

    # print("Train df:\n", train_df, "\n\nTest df:\n", test_df)

    X_train = np.array(train_df[features])
    X_test = np.array(test_df[features])
    y_train = np.array(train_df["label"])
    y_test = np.array(test_df["label"])
    y_train_OHE = np.array(train_df[classes])
    y_test_OHE = np.array(test_df[classes])

    print("Training Features Shape:", X_train.shape)
    print("Testing Features Shape:", X_test.shape)
    print("Training Labels Shape:", y_train.shape)
    print("Testing Labels Shape:", y_test.shape)
    print("Training Labels OHE Shape:", y_train_OHE.shape)
    print("Testing Labels OHE Shape:", y_test_OHE.shape)

    return [df, X_train, X_test, y_train, y_test, y_train_OHE, y_test_OHE]


def evaluateClassifier(cl, X_test, y_test) -> None:  # , classes) -> None:
    print("\nEvaluating classifier...")
    y_pred = cl.predict(X_test)
    y_pred_proba = cl.predict_proba(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_rep = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_rep)


def trainNNmodel(
    X_train, y_train, epochs=50, batch_size=10, min_delta=0.001, patience=5
) -> tuple[Sequential, list]:
    model = Sequential()
    model.add(
        tf.keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation="relu")
    )
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(y_train.shape[1], activation="sigmoid"))
    model.summary()

    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Stop the training when there is no improvement in the validation accuracy for 5 consecutive epochs.
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        min_delta=min_delta,
        patience=patience,
        restore_best_weights=True,
    )

    # now we just update our model fit call
    history = model.fit(
        X_train,
        y_train,
        callbacks=[early_stopping_cb],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        shuffle=True,
        verbose=1,
    )
    return (model, history)
