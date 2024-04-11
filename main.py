import functions as fn
import classifiers as cl
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import tensorflow_decision_forests as tfdf

# Dataset parameters
MAX_SUBJECTS = 22
TRAIN_SUBJECTS = None
TEST_SUBJECTS = None

## RF parameters
N_TREES = 100

## Neural network parameters
EPOCHS = 20
BATCH_SIZE = 10
# Early stopping (metric=val_accuracy)
MIN_DELTA = 0.001
PATIENCE = 5


def main():
    df = fn.readDataset(max_subjects=MAX_SUBJECTS)
    data = cl.preprocessData(df, TRAIN_SUBJECTS, TEST_SUBJECTS)

    usr_in = ""
    while usr_in != "y" and usr_in != "n":
        usr_in = input(
            "Do you want to preload models? Type 'y' for yes and 'n' for no and then hit enter.\n"
        ).lower()
    if usr_in == "y":
        classifiers = cl.loadClassifiers()
        [cl.evaluateClassifier(classifier, data) for classifier in classifiers]
    else:
        #### Train classifiers
        ## Neural Network (tf Sequential model)
        print("Training Neural Network classifier...")
        (nn_model, callbacks) = cl.getSequentialModel(data, MIN_DELTA, PATIENCE)
        history = nn_model.fit(
            data["X_train_NN"],
            data["y_train_NN"],
            callbacks=callbacks,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            shuffle=True,
            verbose=1,
        )
        cl.evaluateClassifier(nn_model, data)
        cl.saveModel(nn_model, MAX_SUBJECTS, TRAIN_SUBJECTS, TEST_SUBJECTS)

        ## Random Forest (tensorflow)
        print("Training Random Forest (tf) classifier...")
        rf_model_tf = tfdf.keras.RandomForestModel()
        rf_model_tf.compile(loss="categorical_crossentropy", metrics=["accuracy"])
        # Train the model
        rf_history = rf_model_tf.fit(data["train_ds"])
        cl.evaluateClassifier(rf_model_tf, data)
        cl.saveModel(rf_model_tf, MAX_SUBJECTS, TRAIN_SUBJECTS, TEST_SUBJECTS)

        ## Random Forest (sklearn)
        print("Training Random Forest classifier...")
        rf_classifier = RandomForestClassifier(
            n_estimators=N_TREES, random_state=7
        ).fit(data["X_train"], data["y_train"])
        cl.evaluateClassifier(rf_classifier, data)
        cl.saveModel(rf_classifier, MAX_SUBJECTS, TRAIN_SUBJECTS, TEST_SUBJECTS)

        ## Gaussian Naive Bayes
        print("Training Gaussian Naive Bayes classifier...")
        gnb_classifier = GaussianNB().fit(data["X_train"], data["y_train"])
        cl.evaluateClassifier(gnb_classifier, data)
        cl.saveModel(gnb_classifier, MAX_SUBJECTS, TRAIN_SUBJECTS, TEST_SUBJECTS)


if __name__ == "__main__":
    main()
