from data_visualization import display_confusion_matrix
import numpy as np
import tensorflow_decision_forests as tfdf
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input
from keras.losses import CategoricalFocalCrossentropy
from keras.optimizers import Adam
from functions import get_label_distribution
from data_visualization import plot_distribution_histograms

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
    print_important_classification_metrics(y_true, y_pred, y_pred_proba)


def print_important_classification_metrics(y_true, y_pred, y_pred_proba=None):
    # Calculate evaluation metrics
    if y_pred_proba is not None:
        auc = roc_auc_score(y_true, y_pred_proba, multi_class="ovr")
        print(f"AUC: {auc:.2f}")
    accuracy = accuracy_score(y_true, y_pred)
    class_rep = classification_report(
        y_true,
        y_pred,
        target_names=[LABEL_LIST[idx] for idx in list(np.unique(y_pred))],
        zero_division=0,
    )
    print(f"Accuracy: {accuracy:.2f}")
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
        loss=CategoricalFocalCrossentropy(alpha=alpha, gamma=4),
        metrics=["accuracy"],
    )

    return model


def train_nn_classifier(
    nn_model: Sequential, X_train, y_train, epochs, batch_size, validation_data=None
):
    """Trains a sequential neural network classifier. Returns training history."""
    
    # Train the neural network on input data
    nn_history = nn_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.3,
        shuffle=True,
        verbose=1,
        validation_data=validation_data,
    )
    return nn_history


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


def custom_loo_cv(X, y, epochs, batch_size):

    # Initialize storage for true and predicted labels
    y_true_total = np.array([], dtype=int)
    y_pred_total = np.array([], dtype=int)
    y_pred_proba_total = np.empty((0, 12), dtype=float)

    # Leave-One-Out Cross-Validation
    for i in range(len(X)):
        # Prepare the training and test sets
        X_train = np.concatenate([X[j] for j in range(len(X)) if j != i], axis=0)
        y_train = np.concatenate([y[j] for j in range(len(y)) if j != i], axis=0)
        X_test, y_test = shuffle(X[i], y[i], random_state=7)

        # Plot train and test label distributions
        y_train_distr = get_label_distribution(y_train)
        y_test_distr = get_label_distribution(y_test)
        plot_distribution_histograms([y_train_distr, y_test_distr])

        # Initialize and train the classifier
        num_classes = np.amax(y_train) + 1
        alpha = calculate_alpha(y_train_distr)
        y_train = np.eye(num_classes)[y_train]
        y_test = np.eye(num_classes)[y_test]

        # Load the classifier model and train it
        clf = get_sequential_model(X_train.shape[1:], num_classes, alpha)
        clf.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        # Make predictions on the test set
        y_pred_proba = clf.predict(X_test)
        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Save true and predicted labels
        y_true_total = np.concatenate([y_true_total, y_true])
        y_pred_total = np.concatenate([y_pred_total, y_pred])
        y_pred_proba_total = np.concatenate([y_pred_proba_total, y_pred_proba])

        # Collect and print classification report
        print(f"Classification report for subject {i}:")
        print_important_classification_metrics(y_true, y_pred)

    # Collect and print total classification report
    print("Total classification report:")
    print_important_classification_metrics(
        y_true_total, y_pred_total, y_pred_proba_total
    )

    return y_true_total, y_pred_total, y_pred_proba_total
