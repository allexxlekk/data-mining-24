from data_visualization import display_confusion_matrix
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input
from keras.losses import CategoricalFocalCrossentropy
from keras.optimizers import Adam
from keras.metrics import Precision, Recall, AUC, F1Score
from functions import get_label_distribution
from data_visualization import plot_distribution_histograms, plot_history

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


def print_important_classification_metrics(
    y_true, y_pred, y_pred_proba=None, filename=""
):
    # Calculate evaluation metrics
    if y_pred_proba is not None:
        auc = roc_auc_score(y_true, y_pred_proba, multi_class="ovr")
        print(f"AUC: {auc:.2f}")
    accuracy = accuracy_score(y_true, y_pred)
    class_rep = classification_report(
        y_true,
        y_pred,
        # labels=LABEL_LIST,
        target_names=LABEL_LIST,  # [LABEL_LIST[idx] for idx in list(np.unique(y_pred))],
        zero_division=0,
    )
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", class_rep)
    display_confusion_matrix(y_true, y_pred, None, LABEL_LIST, filename)


def get_sequential_model(input_shape, output_shape, alpha=0.25) -> Sequential:
    """Returns a compiled sequential model for classification."""

    model = Sequential()
    model.add(Input(shape=(input_shape)))
    model.add(LSTM(20))
    model.add(Dropout(0.1))
    model.add(Dense(output_shape, activation="softmax"))
    model.summary()
    model.compile(
        optimizer=Adam(learning_rate=0.005),
        loss=CategoricalFocalCrossentropy(alpha=alpha, gamma=4),
        metrics=["accuracy", Precision(), Recall(), AUC(), F1Score()],
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


def flatten_array(X: np.array) -> np.array:
    num_samples, num_timesteps, num_sensors = X.shape
    return X.reshape(num_samples, num_timesteps * num_sensors)


def custom_loo_cv(
    cl_type,
    X,
    y,
    epochs=None,
    batch_size=None,
    n_trees=None,
    undersample=True,
    us_factor=4,
):
    """Given a model type (Sequential/RandomForest/BayesianNetworks), train and evaluate the model using Leave-One-Out Cross-Validation (all but one individuals are used during training and the remaining one is used for testing). Finally, the total classification report and the confusion matrix are displayed."""

    # Initialize storage for true and predicted labels
    y_true_total = np.array([], dtype=int)
    y_pred_total = np.array([], dtype=int)
    y_pred_proba_total = np.empty((0, 12), dtype=float)

    # Leave-One-Out Cross-Validation
    for i in range(len(X)):
        print(f"Trainning model {i}...")
        # Prepare the training and test sets
        X_train = np.concatenate([X[j] for j in range(len(X)) if j != i], axis=0)
        y_train = np.concatenate([y[j] for j in range(len(y)) if j != i], axis=0)
        if undersample:
            values, value_counts = np.unique(y_train, return_counts=True)
            scaled_value_counts = scale_array(value_counts, us_factor)
            X_train, y_train = custom_undersample(
                X_train, y_train, dict(zip(values, scaled_value_counts))
            )
        X_test, y_test = shuffle(X[i], y[i], random_state=7)

        # Plot train and test label distributions
        y_train_distr = get_label_distribution(y_train)
        # y_test_distr = get_label_distribution(y_test)
        # plot_distribution_histograms([y_train_distr, y_test_distr])

        if cl_type == "Sequential":
            # Initialize and train the classifier
            num_classes = np.amax(y_train) + 1
            alpha = calculate_alpha(y_train_distr)
            y_train = np.eye(num_classes)[y_train]
            y_test = np.eye(num_classes)[y_test]
            model = get_sequential_model(X_train.shape[1:], num_classes, alpha)

            # Load the classifier model and train it
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

            y_pred_proba = model.predict(X_test)
            y_true = np.argmax(y_test, axis=1)
        else:
            # Flatten the input data
            X_train = flatten_array(X_train)
            X_test = flatten_array(X_test)
            if cl_type == "RandomForest":
                model = RandomForestClassifier(
                    n_estimators=n_trees,
                    random_state=7,
                    n_jobs=-1,
                    verbose=2,
                    # class_weight="balanced",
                    max_depth=10,
                )
            else:
                model = GaussianNB()
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)
            y_true = y_test

        y_pred = np.argmax(y_pred_proba, axis=1)

        # Save true and predicted labels
        y_true_total = np.concatenate([y_true_total, y_true])
        y_pred_total = np.concatenate([y_pred_total, y_pred])
        y_pred_proba_total = np.concatenate([y_pred_proba_total, y_pred_proba])

        # Collect and print classification report
        # print(f"Classification report for subject {i}:")
        # print_important_classification_metrics(y_true, y_pred)

    # Collect and print total classification report
    print("Total classification report:")
    print_important_classification_metrics(
        y_true_total, y_pred_total, y_pred_proba_total, filename=cl_type
    )

    return y_true_total, y_pred_total, y_pred_proba_total


def scale_array(original_array, factor):
    """Scale the values of a 1-d numpy array to a new range defined by a factor of its minimum value."""

    old_min = np.min(original_array)
    old_max = np.max(original_array)
    new_max = old_min * factor

    scaled_array = ((original_array - old_min) / (old_max - old_min)) * (
        new_max - old_min
    ) + old_min
    return scaled_array.astype(int)


def custom_undersample(X, y, sample_count):
    """Undersample the dataset based on the specified sample counts for each class."""

    unique_classes = np.unique(y)
    resampled_indices = []

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        num_samples = sample_count.get(cls, len(cls_indices))

        if num_samples > len(cls_indices):
            raise ValueError(
                f"Requested more samples ({num_samples}) than available ({len(cls_indices)}) for class {cls}"
            )

        # Randomly sample indices without replacement
        sampled_indices = np.random.choice(cls_indices, size=num_samples, replace=False)
        resampled_indices.extend(sampled_indices)

    # Shuffle the resampled indices to mix the classes
    np.random.shuffle(resampled_indices)

    # Create the resampled X and y
    X_resampled = X[resampled_indices]
    y_resampled = y[resampled_indices]

    return X_resampled, y_resampled


def train_and_evaluate_models_no_cv(
    X: list[np.array],
    y: list[np.array],
    epochs=20,
    batch_size=1_000,
    n_trees=100,
    undersample=True,
    us_factor=4,
):
    """Train and evaluate classifiers without using Leave-One-Out Cross-Validation"""

    # Concatenate the data lists into 2 numpy arrays
    X = np.concatenate([X_ind for X_ind in X], axis=0)
    y = np.concatenate([y_ind for y_ind in y], axis=0)

    # Split the dataset to train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7, shuffle=True
    )

    # Perform undersampling on training data
    if undersample:
        # Get original label distribution
        y_train_original_distr = get_label_distribution(y_train)

        # Calculate the number of labels
        values, value_counts = np.unique(y_train, return_counts=True)

        # Calculate the desired number of labels depending on us_factor
        scaled_value_counts = scale_array(value_counts, us_factor)

        # Select calculated desired samples per label
        X_train, y_train = custom_undersample(
            X, y, dict(zip(values, scaled_value_counts))
        )
        # Get label distribution after undersampling
        y_train_us_distr = get_label_distribution(y_train)

        # Print before and after undersampling distributions
        plot_distribution_histograms(
            [y_train_original_distr, y_train_us_distr],
            set_names=["Before undersampling", "After undersampling"],
        )

    #### Neural Network (tf Sequential model)
    print("Training Neural Network classifier...")
    num_classes = np.max(y_train) + 1
    y_train_distr = get_label_distribution(y_train)
    alpha = calculate_alpha(y_train_distr)
    nn_model = get_sequential_model((X_train.shape[1:]), num_classes, alpha)
    nn_history = train_nn_classifier(
        nn_model,
        X_train,
        np.eye(num_classes)[y_train],
        epochs,
        batch_size,
    )
    plot_history(nn_history)
    evaluateClassifier(nn_model, X_test, np.eye(num_classes)[y_test])

    #### Random Forest (sklearn)
    print("Training Random Forest classifier...")
    # Flatten the input data
    X_train = flatten_array(X_train)
    X_test = flatten_array(X_test)

    rf_classifier = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=7,
        n_jobs=-1,
        verbose=2,
        # class_weight="balanced",
        max_depth=5,
    )
    rf_classifier.fit(X_train, y_train)
    evaluateClassifier(
        rf_classifier,
        X_test,
        y_test,
    )

    #### Gaussian Naive Bayes
    print("Training Gaussian Naive Bayes classifier...")
    gnb_classifier = GaussianNB().fit(X_train, y_train)
    evaluateClassifier(gnb_classifier, X_test, y_test)
