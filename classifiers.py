import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
import tensorflow as tf
from keras.models import Sequential
import tensorflow_decision_forests as tfdf
from sklearn.ensemble import RandomForestClassifier
import joblib
from os.path import exists
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

LABEL_LIST = {
    "walking",
    "running",
    "shuffling",
    "stairs (ascending)",
    "stairs (descending)",
    "standing",
    "sitting",
    "lying",
    "cycling (sit)",
    "cycling (stand)",
    "cycling (sit, inactive)",
    "cycling (stand, inactive)",
}

FEATURES = [
    "ID",
    # "time_step",
    "month",
    "day",
    "hour_cos",
    "hour_sin",
    "minute_cos",
    "minute_sin",
    "day_of_week_cos",
    "day_of_week_sin",
    "back_x",
    "back_y",
    "back_z",
    "thigh_x",
    "thigh_y",
    "thigh_z",
]


def preprocessData(df: pd.DataFrame, train_subjects=None, test_subjects=None) -> dict:
    """Prepares inputs (features and labels) for trainning and testing classifiers."""

    # Standardize numerical features
    numerical_columns = FEATURES.copy()
    numerical_columns.remove("ID")
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    data = {}

    # One-hot encode the label and ID columns
    labels_OHE = pd.get_dummies(df["label"], dtype=float, prefix="label")
    id_OHE = pd.get_dummies(df["ID"], dtype=float, prefix="id")

    # Concatenate one-hot encoded columns with the original dataframe
    data["df"] = pd.concat([df, labels_OHE, id_OHE], axis=1)

    label_columns = labels_OHE.columns.tolist()
    id_columns = id_OHE.columns.tolist()

    if train_subjects != None and test_subjects != None:
        unique_subject_ids = df["ID"].unique()
        if train_subjects + test_subjects > len(unique_subject_ids):
            print(
                f"Train and test subjects exceed total subjects (22)! Setting train subjects to 17 and test subjects to 5."
            )
            train_subjects = 17
            test_subjects = 5

        # Seperate the data into train and test subjects based on user's choice
        first_ids_df = data["df"][
            data["df"]["ID"].isin(unique_subject_ids[:train_subjects])
        ]  # .drop(labels="ID", axis=1)
        next_ids_df = data["df"][
            data["df"]["ID"].isin(
                unique_subject_ids[train_subjects : train_subjects + test_subjects]
            )
        ]  # .drop(labels="ID", axis=1)
        # Shuffle dataframe rows
        data["train_df"] = first_ids_df.sample(frac=1, random_state=7).reset_index(
            drop=True
        )
        data["test_df"] = next_ids_df.sample(frac=1, random_state=7).reset_index(
            drop=True
        )
    else:
        # Shuffle dataframe rows
        data["df"] = data["df"].sample(frac=1).reset_index(drop=True)
        # Calculate the index to split the dataframe
        split_index = int(0.7 * len(data["df"]))
        # Split the dataframe into train and test sets
        data["train_df"] = data["df"].iloc[:split_index]
        data["test_df"] = data["df"].iloc[split_index:]

    # print("Train df:\n", train_df, "\n\nTest df:\n", test_df)

    nn_features = FEATURES.copy() + id_columns
    nn_features.remove("ID")
    data["nn_features"] = nn_features
    # print(nn_features, len(nn_features))

    data["X_train"] = np.array(data["train_df"][FEATURES])
    data["X_test"] = np.array(data["test_df"][FEATURES])
    data["X_train_NN"] = np.array(data["train_df"][nn_features])
    data["X_test_NN"] = np.array(data["test_df"][nn_features])
    data["y_train"] = np.array(data["train_df"]["label"])
    data["y_test"] = np.array(data["test_df"]["label"])
    data["y_train_NN"] = np.array(data["train_df"][label_columns])
    data["y_test_NN"] = np.array(data["test_df"][label_columns])

    print("Training Features Shape:", data["X_train"].shape)
    print("Testing Features Shape:", data["X_test"].shape)
    print("Training NN Features Shape:", data["X_train_NN"].shape)
    print("Testing NN Features Shape:", data["X_test_NN"].shape)
    print("Training Labels Shape:", data["y_train"].shape)
    print("Testing Labels Shape:", data["y_test"].shape)
    print("Training Labels OHE Shape:", data["y_train_NN"].shape)
    print("Testing Labels OHE Shape:", data["y_test_NN"].shape)

    return data


def evaluateClassifier(
    cl: tfdf.keras.RandomForestModel | RandomForestClassifier | Sequential, data: dict
) -> None:  # , classes) -> None:
    print(f"\nEvaluating {type(cl)} classifier...")
    if type(cl) == RandomForestClassifier:
        y_pred = cl.predict(data["X_test"])
        y_pred_proba = cl.predict_proba(data["X_test"])
    else:
        if type(cl) == tfdf.keras.RandomForestModel:
            test_df = data["test_df"][data["nn_features"] + ["label"]]
            test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="label")
            y_pred_proba = cl.predict(test_ds)
        else:
            y_pred_proba = cl.predict(data["X_test_NN"])
        y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate evaluation metrics
    accuracy = accuracy_score(data["y_test"], y_pred)
    auc = roc_auc_score(data["y_test_NN"], y_pred_proba, multi_class="ovr")
    conf_matrix = confusion_matrix(data["y_test"], y_pred)
    class_rep = classification_report(
        data["y_test"], y_pred, target_names=LABEL_LIST, zero_division=0
    )

    print(f"Accuracy: {accuracy:.2f}")
    print(f"AUC: {auc:.2f}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_rep)


def trainNNmodel(
    data, epochs=50, batch_size=10, min_delta=0.001, patience=5
) -> tuple[Sequential, list]:
    model = Sequential()
    model.add(
        tf.keras.layers.Dense(
            64, input_shape=(data["X_train_NN"].shape[1],), activation="relu"
        )
    )
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(data["y_train_NN"].shape[1], activation="sigmoid"))
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
        data["X_train_NN"],
        data["y_train_NN"],
        callbacks=[early_stopping_cb],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        shuffle=True,
        verbose=1,
    )
    return (model, history)


def trainRFClassifier_tf(data: dict) -> tuple[tfdf.keras.RandomForestModel, list]:
    # tf_ds_cols = FEATURES + ["label"]
    tf_ds_cols = data["nn_features"] + ["label"]
    train_df = data["train_df"][tf_ds_cols]
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="label")

    model = tfdf.keras.RandomForestModel()
    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(train_ds)

    return (model, history)


def getModelPath(
    cl_type,
    max_subjects=23,
    train_subjects=None,
    test_subjects=None,
    models_path="models",
):
    if cl_type.__name__ == "RandomForestClassifier":
        extension = "joblib"
        model_class = "sklearn_RF"
    else:
        extension = "keras"
        if cl_type.__name__ == "Sequential":
            model_class = "keras_NN"
        else:
            model_class = "keras_RF"

    split_str = (
        f"{train_subjects}_{test_subjects}"
        if train_subjects is not None and test_subjects is not None
        else "no"
    )

    path = f"{models_path}/{model_class}_classifier_{max_subjects}_max_subjects_{split_str}_split.{extension}"

    return path


def saveModel(
    cl: tfdf.keras.RandomForestModel | RandomForestClassifier | Sequential,
    max_subjects,
    train_subjects=None,
    test_subjects=None,
    models_path="models",
) -> None:
    """Generates path to save the trained model and saves it."""

    cl_type = type(cl)
    path = getModelPath(
        cl_type, max_subjects, train_subjects, test_subjects, models_path
    )

    if not exists(path):
        if cl_type == RandomForestClassifier:
            joblib.dump(cl, path)
        else:
            cl.save(path)
    else:
        print(f"File {path} already exists!")


def loadClassifiers(
    cl_types: list = [
        tfdf.keras.RandomForestModel,
        RandomForestClassifier,
        Sequential,
    ],
    models_path="models/only_sensors",
    max_subjects=23,
    train_subjects=None,
    test_subjects=None,
) -> list:
    classifiers = []
    for cl_type in cl_types:
        print(f"Looking for classifier of type: {cl_type.__name__}...")
        path = getModelPath(
            cl_type, max_subjects, train_subjects, test_subjects, models_path
        )
        if exists(path):
            print(f"Found classifier {path}. Adding it to the classifiers list.")
            if cl_type.__name__ == "RandomForestClassifier":
                classifiers.append(joblib.load(path))
            else:
                print(cl_type.__name__)
                classifiers.append(load_model(path))
        else:
            print(f"Didn't find classifier in path: {path}.")

    return classifiers
