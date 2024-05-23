import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import functions as fn
import classifiers as cl
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import data_visualization as dv
import numpy as np

## Dataset parameters
MAX_SUBJECTS = 8
TRAIN_SUBJECTS = None
TEST_SUBJECTS = None

## RF parameters
N_TREES = 100

## Neural network parameters
EPOCHS = 2
BATCH_SIZE = 10_000


def main():
    X_train, X_test, y_train, y_test = fn.readAndPreprocessData(
        max_subjects=MAX_SUBJECTS,
        train_subjects=TRAIN_SUBJECTS,
        test_subjects=TEST_SUBJECTS,
        window_size=20,
    )

    dv.plotLabelDistributionHistogram(y_train, y_test)

    usr_in = "n"
    while usr_in != "y" and usr_in != "n":
        usr_in = input(
            "Do you want to preload models? Type 'y' for yes and 'n' for no and then hit enter.\n"
        ).lower()
    if usr_in == "y":
        classifiers = cl.loadClassifiers()
        [
            cl.evaluateClassifier(classifier, X_test, y_test)
            for classifier in classifiers
        ]
    else:
        #### Train classifiers
        # Neural Network (tf Sequential model)
        print("Training Neural Network classifier...")
        # Determine the number of unique classes
        num_classes = np.max(y_train) + 1
        # Convert labels to OHE labels
        y_train = np.eye(num_classes)[y_train]
        nn_model = cl.getSequentialModel((X_train.shape[1:]), num_classes)
        nn_history = cl.trainNNClassifier(
            nn_model, X_train, y_train, EPOCHS, BATCH_SIZE
        )
        dv.plotHistory(nn_history)
        cl.evaluateClassifier(nn_model, X_test, np.eye(num_classes)[y_test])
        # cl.saveModel(nn_model, MAX_SUBJECTS, TRAIN_SUBJECTS, TEST_SUBJECTS)

        # Random Forest (sklearn)
        print("Training Random Forest classifier...")
        # Flatten the input data
        num_samples, num_timesteps, num_sensors = X_train.shape
        X_train = X_train.reshape(num_samples, num_timesteps * num_sensors)
        num_samples, num_timesteps, num_sensors = X_test.shape
        X_test = X_test.reshape(num_samples, num_timesteps * num_sensors)
        rf_classifier = RandomForestClassifier(
            n_estimators=N_TREES,
            random_state=7,
            n_jobs=-1,
            verbose=3,
            class_weight="balanced",
        ).fit(X_train, y_train)
        cl.evaluateClassifier(rf_classifier, X_test, y_test)
        # cl.saveModel(rf_classifier, MAX_SUBJECTS, TRAIN_SUBJECTS, TEST_SUBJECTS)

        ## Gaussian Naive Bayes
        # print("Training Gaussian Naive Bayes classifier...")
        # gnb_classifier = GaussianNB().fit(X_train, y_train_OHE)
        # cl.evaluateClassifier(gnb_classifier, data)
        # cl.saveModel(gnb_classifier, MAX_SUBJECTS, TRAIN_SUBJECTS, TEST_SUBJECTS)


if __name__ == "__main__":
    main()
