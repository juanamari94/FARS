import csv
import os

from numpy import genfromtxt
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection

from utils.data_parser import parse_datasets

rootdir = os.path.dirname(os.path.realpath(__file__))


def format_training_set(dataset):
    features = dataset[:, :len(dataset[0]) - 1]
    labels = dataset[:, len(dataset[0]) - 1]
    return features, labels


def train_model():
    train_features, train_labels = format_training_set(
        genfromtxt(os.path.join(rootdir, "data", "parsed_data", "fars_train.csv"), delimiter=","))
    X_train, X_cv, y_train, y_cv = model_selection.train_test_split(train_features, train_labels)
    model = linear_model.LogisticRegression(multi_class="ovr")
    model.fit(X_train, y_train)
    cv_predictions = model.predict(X_cv)
    print("CV set has an accuracy score of: {}".format(metrics.accuracy_score(y_cv, cv_predictions)))
    return model


def predict_test_set(model):
    test_features = genfromtxt(os.path.join(rootdir, "data", "parsed_data", "fars_test.csv"),
                               delimiter=",")  # We don't have labels for this one
    classification_results = model.predict(test_features)
    return classification_results


def output_results(results):
    result_dir = os.path.join(rootdir, "result")
    if not os.path.exists(os.path.join(rootdir, "result")):
        os.makedirs(result_dir)

    with open(os.path.join(result_dir, "test_set_predictions_output.csv"), "w+") as output:
        csvwriter = csv.writer(output, delimiter=",")
        for result in results:
            csvwriter.writerow([int(result)])


if __name__ == "__main__":
    parse_datasets()
    print("Using a Logistic Regression with OneVsRest or OneVsAll methodology for multi-class classification.")
    print("Training model...")
    model = train_model()
    print("Done!")
    print("Classifying test set...")
    results = predict_test_set(model)
    print("Done!")
    print("Creating output file...")
    output_results(results)
    print("Done! Output will be in {}".format(os.path.join(rootdir, "result", "test_set_predictions_output.csv")))
