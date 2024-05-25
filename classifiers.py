from data_visualization import display_confusion_matrix
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
from keras.optimizers import Adam

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
    cl: RandomForestClassifier | Sequential | GaussianNB,
    X_test: np.array,
    y_test: np.array,
) -> None:
    """Evaluates a model and prints important classification metrics."""

    print(f"\nEvaluating {type(cl)} classifier...")

    if type(cl) == RandomForestClassifier or type(cl) == GaussianNB:
        y_pred_proba = cl.predict_proba(X_test)
        y_true = y_test
    elif type(cl) == Sequential:
        y_pred_proba = cl.predict(X_test)
        y_true = np.argmax(y_test, axis=1)

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
    display_confusion_matrix(y_true, y_pred, None, LABEL_LIST)


def get_sequential_model(input_shape, output_shape, alpha=0.25) -> Sequential:
    """Returns a compiled sequential model for classification."""

    model = Sequential()
    model.add(Input(shape=(input_shape)))
    model.add(LSTM(20))
    model.add(Dropout(0.1))
    model.add(Dense(output_shape, activation="softmax"))
    model.summary()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=CategoricalFocalCrossentropy(alpha=alpha, gamma=2),
        metrics=["accuracy"],
    )

    return model


def train_nn_classifier(nn_model: Sequential, X_train, y_train, epochs, batch_size):
    """Trains a sequential neural network classifier. Returns training history."""

    # Stop the training when there is no improvement for some (patience) consecutive epochs.
    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        min_delta=0.0001,
        patience=20,
        restore_best_weights=True,
    )
    callbacks = [early_stopping_cb]

    # Train the neural network on input data
    nn_history = nn_model.fit(
        X_train,
        y_train,
        callbacks=callbacks,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.3,
        shuffle=True,
        verbose=1,
    )
    return nn_history


def get_model_path(
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


def save_model(
    cl: tfdf.keras.RandomForestModel | RandomForestClassifier | Sequential | GaussianNB,
    max_subjects,
    train_subjects=None,
    test_subjects=None,
    models_path="models",
) -> None:
    """Saves a model to storage."""

    cl_type = type(cl)
    path = get_model_path(
        cl_type, max_subjects, train_subjects, test_subjects, models_path
    )

    if not exists(path):
        if cl_type == RandomForestClassifier or cl_type == GaussianNB:
            joblib.dump(cl, path)
        else:
            cl.save(path)
    else:
        print(f"File {path} already exists!")


def load_classifiers(
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
        path = get_model_path(
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


def calculate_alpha(label_distribution: list[float]) -> list[float]:
    """Calculate the alpha parameter for each label based on the label distribution."""

    # Convert percentages to proportions (sum should be 1)
    total = sum(label_distribution)
    proportions = [x / total for x in label_distribution]

    # Calculate the inverse of the proportions
    inverse_proportions = [1 / p if p > 0 else 0 for p in proportions]

    # Normalize the inverses to sum to 1
    sum_inverse_proportions = sum(inverse_proportions)
    alpha = [x / sum_inverse_proportions for x in inverse_proportions]

    return alpha
