from data_visualization import displayConfusionMatrix
import numpy as np
import joblib
from os.path import exists
import tensorflow_decision_forests as tfdf
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Input
from keras.callbacks import EarlyStopping
from keras.losses import CategoricalFocalCrossentropy

LABEL_LIST = [
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
]


def evaluateClassifier(
    cl: tfdf.keras.RandomForestModel | RandomForestClassifier | Sequential | GaussianNB,
    data: dict,
) -> None:
    """Evaluates a model and prints important classification metrics."""

    print(f"\nEvaluating {type(cl)} classifier...")
    print(f"Features: {data['nn_features']}")

    y_true = data["y_test"]
    if type(cl) == RandomForestClassifier or type(cl) == GaussianNB:
        y_pred_proba = cl.predict_proba(data["X_test"])
    else:
        if type(cl) == tfdf.keras.RandomForestModel:
            y_pred_proba = cl.predict(data["test_ds"])
        else:
            y_pred_proba = cl.predict(data["X_test_RNN"])
            y_true = np.argmax(data["y_test_RNN"], axis=1)

    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba, multi_class="ovr")
    class_rep = classification_report(
        y_true, y_pred, target_names=LABEL_LIST, zero_division=0
    )
    print(f"Accuracy: {accuracy:.2f}")
    print(f"AUC: {auc:.2f}")
    print("Classification Report:\n", class_rep)
    displayConfusionMatrix(y_true, y_pred, None, LABEL_LIST)


def getSequentialModel(input_shape, output_shape) -> Sequential:
    """Returns a compiled sequential model and its training callbacks."""

    model = Sequential()
    model.add(Input(shape=(input_shape)))
    model.add(LSTM(20, dropout=0.1))
    model.add(Dropout(0.1))
    model.add(Dense(output_shape, activation="softmax"))
    model.summary()
    model.compile(
        optimizer="Adam", loss=CategoricalFocalCrossentropy(), metrics=["accuracy"]
    )

    return model


def getModelPath(
    cl_type,
    max_subjects=23,
    train_subjects=None,
    test_subjects=None,
    models_path="models",
):
    """Generates path to save the trained model based on its type and other variables."""

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
    """Saves a model to storage."""

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
    """Loads all available saved classifiers from storage."""

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
                classifiers.append(load_model(path))
        else:
            print(f"Didn't find classifier in path: {path}.")

    return classifiers


def trainNNClassifier(nn_model: Sequential, data: dict, epochs, batch_size):
    # Stop the training when there is no improvement for some (patience) consecutive epochs.
    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        min_delta=0.001,
        patience=5,
        restore_best_weights=True,
    )
    callbacks = [early_stopping_cb]
    nn_history = nn_model.fit(
        data["X_train_RNN"],
        data["y_train_RNN"],
        callbacks=callbacks,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        shuffle=True,
        verbose=1,
    )
    return nn_history
