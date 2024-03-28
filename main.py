import functions as fn
import classifiers as cl
from sklearn.ensemble import RandomForestRegressor

N_TREES = 500


def main():
    df = fn.readDataset(max_subjects=1)
    [df, X_train, X_test, y_train, y_test, classes] = cl.preprocessData(df)

    ######## Train classifiers on train data and then evaluate them on test Data

    #### Random Forest Classifier
    print("Training Random Forest classifier...")
    rf_classifier = RandomForestRegressor(n_estimators=N_TREES, random_state=7)
    rf_classifier.fit(X_train, y_train)
    cl.evaluateClassifier(rf_classifier, X_test, y_test, classes)

    #### Bayesian Networks Classifier

    #### Neural Network Classifier


if __name__ == "__main__":
    main()
