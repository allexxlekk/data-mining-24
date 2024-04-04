import functions as fn
import classifiers as cl
from sklearn.ensemble import RandomForestClassifier

# Dataset parameters
MAX_SUBJECTS = 23
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

    [
        df,
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_OHE,
        y_test_OHE,
        train_df,
        test_df,
    ] = cl.preprocessData_v2(df, TRAIN_SUBJECTS, TEST_SUBJECTS)

    usr_in = ""
    while usr_in != "y" and usr_in != "n":
        usr_in = input(
            "Do you want to preload models? Type 'y' for yes and 'n' for no and then hit enter."
        )
    if usr_in == "y":
        classifiers = cl.loadClassifiers()
    else:
        #### Train classifiers
        ## Random Forest (tensorflow)
        print("Training Random Forest (tf) classifier...")
        (rf_model_tf, rf_history) = cl.trainRFClassifier_tf(train_df)

        ## Random Forest (sklearn)
        print("Training Random Forest classifier...")
        rf_classifier = RandomForestClassifier(n_estimators=N_TREES, random_state=7)
        rf_classifier = rf_classifier.fit(X_train, y_train)

        ## Neural Network
        print("Training Neural Network classifier...")
        (nn_model, nn_history) = cl.trainNNmodel(
            X_train,
            y_train_OHE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            min_delta=MIN_DELTA,
            patience=PATIENCE,
        )

        classifiers = [nn_model, rf_model_tf, rf_classifier]

    for classifier in classifiers:
        print(f"\Saving {type(classifier)} model...")
        cl.saveModel(
            classifier,
            max_subjects=MAX_SUBJECTS,
            train_subjects=TRAIN_SUBJECTS,
            test_subjects=TEST_SUBJECTS,
        )
        print(f"\nEvaluating {type(classifier)} model...")
        cl.evaluateClassifier(
            classifier,
            X_test=X_test,
            y_test=y_test,
            test_df=test_df,
            y_test_OHE=y_test_OHE,
        )


if __name__ == "__main__":
    main()
