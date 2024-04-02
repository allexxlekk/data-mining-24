import functions as fn
import classifiers as cl
import joblib
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model

## RF settings
N_TREES = 100

## Neural network settings
EPOCHS = 20
BATCH_SIZE = 10
# Early stopping (metric=val_accuracy)
MIN_DELTA = 0.001
PATIENCE = 5


def main():
    df = fn.readDataset(max_subjects=2)

    [df, X_train, X_test, y_train, y_test, y_train_OHE, y_test_OHE] = (
        cl.preprocessData_v2(df)
    )
    print("Saving preprocessed dataframe to file...")
    df.to_csv("output/df.csv", index=False)

    ######## Train classifiers on train data and then evaluate them on test Data

    print("Training Random Forest classifier...")
    rf_classifier = RandomForestClassifier(n_estimators=N_TREES, random_state=7)
    rf_classifier.fit(X_train, y_train)
    joblib.dump(rf_classifier, "models/rf_classifier.joblib")
    cl.evaluateClassifier(rf_classifier, X_test, y_test)

    #### Neural Network Classifier
    print("Training Neural Network classifier...")
    (nn_model, history) = cl.trainNNmodel(
        X_train, y_train_OHE, EPOCHS, BATCH_SIZE, MIN_DELTA, PATIENCE
    )
    nn_model.evaluate(X_test, y_test_OHE)
    nn_model.save("models/nn_model.keras")

    #### Bayesian Networks Classifier

    ## Load pre-trained classifiers
    rf_classifier = joblib.load("models/rf_classifier.joblib")
    nn_model = load_model("models/nn_model.keras")


if __name__ == "__main__":
    main()
