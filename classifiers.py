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
    "time_step",
    "back_x",
    "back_y",
    "back_z",
    "thigh_x",
    "thigh_y",
    "thigh_z",
]


def preprocessData_v2(
    df: pd.DataFrame, train_subjects=None, test_subjects=None
) -> list:
    """Prepares inputs (features and labels) for trainning and testing classifiers."""

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
        train_df = first_ids_df.sample(frac=1, random_state=7).reset_index(drop=True)
        test_df = next_ids_df.sample(frac=1, random_state=7).reset_index(drop=True)
    else:
        # Shuffle dataframe rows
        df = df.sample(frac=1).reset_index(drop=True)
        # Calculate the index to split the dataframe
        split_index = int(0.7 * len(df))
        # Split the dataframe into train and test sets
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]

    # print("Train df:\n", train_df, "\n\nTest df:\n", test_df)

    X_train = np.array(train_df[FEATURES])
    X_test = np.array(test_df[FEATURES])
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

    return [
        df,
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_OHE,
        y_test_OHE,
        train_df,
        test_df,
    ]


def evaluateClassifier(
    cl: tfdf.keras.RandomForestModel | RandomForestClassifier | Sequential,
    X_test=None,
    y_test=None,
    y_test_OHE=None,
    test_df=None,
) -> None:  # , classes) -> None:
    print(f"\nEvaluating {type(cl)} classifier...")
    if type(cl) == RandomForestClassifier:
        y_pred = cl.predict(X_test)
        y_pred_proba = cl.predict_proba(X_test)
    else:
        if type(cl) == tfdf.keras.RandomForestModel:
            test_df = test_df[FEATURES + ["label"]]
            X_test = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="label")
        y_pred_proba = cl.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test_OHE, y_pred_proba, multi_class="ovr")
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_rep = classification_report(
        y_test, y_pred, target_names=LABEL_LIST, zero_division=0
    )

    print(f"Accuracy: {accuracy:.2f}")
    print(f"AUC: {auc:.2f}")
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


def trainRFClassifier_tf(train_df) -> tuple[tfdf.keras.RandomForestModel, list]:
    train_df = train_df[FEATURES + ["label"]]
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
        print(cl_type.__name__)
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

    return classifiers
