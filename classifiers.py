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
from sklearn.naive_bayes import GaussianNB

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
    # "ID",
    # "time_step",
    # "year",
    # "month",
    # "day",
    # "hour_cos",
    # "hour_sin",
    # "minute_cos",
    # "minute_sin",
    # "day_of_week_cos",
    # "day_of_week_sin",
    "back_x",
    "back_y",
    "back_z",
    "thigh_x",
    "thigh_y",
    "thigh_z",
]


def preprocessData(df: pd.DataFrame, train_subjects=None, test_subjects=None) -> dict:
    """Prepares inputs (features and labels) for trainning and testing classifiers."""

    # One-hot encode the label and ID columns
    labels_OHE = pd.get_dummies(df["label"], dtype=float, prefix="label")
    id_OHE = pd.get_dummies(df["ID"], dtype=float, prefix="id")
    label_columns = labels_OHE.columns.tolist()

    # Standardize numerical features
    numerical_columns = FEATURES.copy()
    nn_features = FEATURES.copy()
    if "ID" in FEATURES:
        numerical_columns.remove("ID")
        nn_features.remove("ID")
        id_columns = id_OHE.columns.tolist()
        nn_features += id_columns

    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Concatenate one-hot encoded columns with the original dataframe
    data = {}
    data["df"] = pd.concat([df, labels_OHE, id_OHE], axis=1)

    if train_subjects != None and test_subjects != None:
        unique_subject_ids = df["ID"].unique()
        if train_subjects + test_subjects > len(unique_subject_ids):
            print(
                f"Train and test subjects exceed total subjects {len(unique_subject_ids)}!"
            )
            train_subjects = int(len(unique_subject_ids) * 2 / 3)
            test_subjects = int(len(unique_subject_ids) * 1 / 3)
            print(
                f"Setting train subjects to {train_subjects} and subjects to {test_subjects}"
            )

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
        data["df"] = data["df"].sample(frac=1).reset_index(drop=True)
        # Calculate the index to split the dataframe
        split_index = int(0.7 * len(data["df"]))
        # Split the dataframe into train and test sets
        data["train_df"] = data["df"].iloc[:split_index]
        data["test_df"] = data["df"].iloc[split_index:]

    data["nn_features"] = nn_features
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

    column_sums_train = data["train_df"][label_columns].sum()
    total_sum_train = column_sums_train.sum()
    column_sum_train_percentages = round((column_sums_train / total_sum_train) * 100, 2)

    column_sums_test = data["test_df"][label_columns].sum()
    total_sum_test = column_sums_test.sum()
    column_sum_test_percentages = round((column_sums_test / total_sum_test) * 100, 2)

    print("Train df label distribution:")
    print(column_sum_train_percentages)
    print("Test df label distribution:")
    print(column_sum_test_percentages)

    return data


def evaluateClassifier(
    cl: tfdf.keras.RandomForestModel | RandomForestClassifier | Sequential | GaussianNB,
    data: dict,
) -> None:  # , classes) -> None:
    print(f"\nEvaluating {type(cl)} classifier...")
    print(f"Features: {data['nn_features']}")
    if type(cl) == RandomForestClassifier or type(cl) == GaussianNB:
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
    elif cl_type.__name__ == "GaussianNB":
        extension = "joblib"
        model_class = "sklearn_GNB"
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
    cl: tfdf.keras.RandomForestModel | RandomForestClassifier | Sequential | GaussianNB,
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
        if cl_type == RandomForestClassifier or cl_type == GaussianNB:
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
        GaussianNB,
    ],
    models_path="models",
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
