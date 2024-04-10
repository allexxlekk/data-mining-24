import functions as fn
import classifiers as cl
from sklearn.ensemble import RandomForestClassifier

# Dataset parameters
MAX_SUBJECTS = 22
TRAIN_SUBJECTS = 12
TEST_SUBJECTS = 6

## RF parameters
N_TREES = 100

## Neural network parameters
EPOCHS = 10
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
        (nn_model, nn_history) = cl.trainNNmodel(
            data,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            min_delta=MIN_DELTA,
            patience=PATIENCE,
        )
        cl.evaluateClassifier(nn_model, data)
        cl.saveModel(nn_model, MAX_SUBJECTS, TRAIN_SUBJECTS, TEST_SUBJECTS)

        ## Random Forest (tensorflow)
        print("Training Random Forest (tf) classifier...")
        (rf_model_tf, rf_history) = cl.trainRFClassifier_tf(data)
        cl.evaluateClassifier(rf_model_tf, data)
        cl.saveModel(rf_model_tf, MAX_SUBJECTS, TRAIN_SUBJECTS, TEST_SUBJECTS)

        ## Random Forest (sklearn)
        print("Training Random Forest classifier...")
        rf_classifier = RandomForestClassifier(n_estimators=N_TREES, random_state=7)
        rf_classifier = rf_classifier.fit(data["X_train"], data["y_train"])
        cl.evaluateClassifier(rf_classifier, data)
        cl.saveModel(rf_classifier, MAX_SUBJECTS, TRAIN_SUBJECTS, TEST_SUBJECTS)


if __name__ == "__main__":
    main()
