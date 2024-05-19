import functions as fn
import classifiers as cl
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import tensorflow_decision_forests as tfdf
import data_visualization as dv
from keras.losses import CategoricalFocalCrossentropy

## Dataset parameters
MAX_SUBJECTS = 22
TRAIN_SUBJECTS = None
TEST_SUBJECTS = None

## RF parameters
N_TREES = 100

## Neural network parameters
EPOCHS = 1
BATCH_SIZE = 256


def main():
    data = fn.readAndPreprocessData(
        max_subjects=MAX_SUBJECTS,
        train_subjects=TRAIN_SUBJECTS,
        test_subjects=TEST_SUBJECTS,
        window_size=25,
    )

    dv.plotLabelDistributionHistogram(
        data["y_train_RNN"], data["y_test_RNN"], cl.LABEL_LIST
    )

    usr_in = "n"
    while usr_in != "y" and usr_in != "n":
        usr_in = input(
            "Do you want to preload models? Type 'y' for yes and 'n' for no and then hit enter.\n"
        ).lower()
    if usr_in == "y":
        classifiers = cl.loadClassifiers()
        [cl.evaluateClassifier(classifier, data) for classifier in classifiers]
    else:
        #### Train classifiers
        # Neural Network (tf Sequential model)
        print("Training Neural Network classifier...")
        nn_model = cl.getSequentialModel(
            (data["X_train_RNN"].shape[1:]), data["y_train_RNN"].shape[1]
        )
        nn_history = cl.trainNNClassifier(nn_model, data, EPOCHS, BATCH_SIZE)
        dv.plotHistory(nn_history)
        cl.evaluateClassifier(nn_model, data)
        # cl.saveModel(nn_model, MAX_SUBJECTS, TRAIN_SUBJECTS, TEST_SUBJECTS)

        # Random Forest (tensorflow)
        print("Training Random Forest (tf) classifier...")
        rf_model_tf = tfdf.keras.RandomForestModel()
        rf_model_tf.compile(loss=CategoricalFocalCrossentropy, metrics=["accuracy"])
        # Train the model
        rf_history = rf_model_tf.fit(data["train_ds"], verbose=1)
        cl.evaluateClassifier(rf_model_tf, data)
        # cl.saveModel(rf_model_tf, MAX_SUBJECTS, TRAIN_SUBJECTS, TEST_SUBJECTS)

        ## Random Forest (sklearn)
        print("Training Random Forest classifier...")
        rf_classifier = RandomForestClassifier(
            n_estimators=N_TREES, random_state=7
        ).fit(data["X_train"], data["y_train"])
        cl.evaluateClassifier(rf_classifier, data)
        # cl.saveModel(rf_classifier, MAX_SUBJECTS, TRAIN_SUBJECTS, TEST_SUBJECTS)

        ## Gaussian Naive Bayes
        print("Training Gaussian Naive Bayes classifier...")
        gnb_classifier = GaussianNB().fit(data["X_train"], data["y_train"])
        cl.evaluateClassifier(gnb_classifier, data)
        # cl.saveModel(gnb_classifier, MAX_SUBJECTS, TRAIN_SUBJECTS, TEST_SUBJECTS)


if __name__ == "__main__":
    main()
