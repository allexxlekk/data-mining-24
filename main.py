import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import functions as fn
import classifiers as cl
import data_visualization as dv
import clustering
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering, KMeans

## Dataset parameters
SUBJECTS = 22
TRAIN_SUBJECTS = None
TEST_SUBJECTS = None

## RF parameters
N_TREES = 100

## Neural network parameters
EPOCHS = 50
BATCH_SIZE = 5_000


def main():
    # X, y = fn.read_and_preprocess_data(subjects=SUBJECTS)
    X, y = fn.load_preprocessed_data()

    #### Add data visualization here
    #
    #
    
    #### Train classifiers

    # Neural Network (tf Sequential model)
    print("Training Neural Network classifier...")
    # cl.custom_loo_cv(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    X = np.concatenate([X[i] for i in range(len(X))], axis=0)
    y = np.concatenate([y[i] for i in range(len(y))], axis=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7, shuffle=True
    )
    num_classes = np.max(y_train) + 1
    y_train_distr = cl.get_label_distribution(y_train)
    alpha = cl.calculate_alpha(y_train_distr)
    nn_model = cl.get_sequential_model((X_train.shape[1:]), num_classes, alpha)
    nn_history = cl.train_nn_classifier(
        nn_model,
        X_train,
        np.eye(num_classes)[y_train],
        EPOCHS,
        BATCH_SIZE,
    )
    dv.plot_history(nn_history)
    cl.evaluateClassifier(nn_model, X_test, np.eye(num_classes)[y_test])

    # Random Forest (sklearn)
    print("Training Random Forest classifier...")
    # Flatten the input data
    num_samples, num_timesteps, num_sensors = X_train.shape
    test_samples = X_test.shape[0]
    X_train = X_train.reshape(num_samples, num_timesteps * num_sensors)
    X_test = X_test.reshape(test_samples, num_timesteps * num_sensors)
    rf_classifier = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=7,
        n_jobs=-1,
        verbose=2,
        class_weight="balanced",
    ).fit(X_train, y_train)
    cl.evaluateClassifier(
        rf_classifier,
        X_test,
        y_test,
    )

    # Gaussian Naive Bayes
    print("Training Gaussian Naive Bayes classifier...")
    gnb_classifier = GaussianNB().fit(X_train, y_train)
    cl.evaluateClassifier(gnb_classifier, X_test, y_test)

    #### Clustering
    # Create feature matrix
    feature_matrix = clustering.create_feature_matrix(y)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=5, random_state=7)
    km_labels = kmeans.fit_predict(feature_matrix)
    clustering.plot_clusters(feature_matrix, km_labels)


    # Perform hierarchical clustering
    agg_clustering = AgglomerativeClustering(n_clusters=5)
    agg_labels = agg_clustering.fit_predict(feature_matrix)
    clustering.plot_dendogram(feature_matrix, agg_labels)
    clustering.plot_clusters(feature_matrix, agg_labels)


if __name__ == "__main__":
    main()
