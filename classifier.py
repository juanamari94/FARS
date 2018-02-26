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
        genfromtxt("{}/data/parsed_data/fars_train.csv".format(rootdir), delimiter=","))
    X_train, X_cv, y_train, y_cv = model_selection.train_test_split(train_features, train_labels)
    model = linear_model.LogisticRegression(multi_class="ovr")
    model.fit(X_train, y_train)
    cv_predictions = model.predict(X_cv)
    print("CV set has an accuracy score of: {}".format(metrics.accuracy_score(y_cv, cv_predictions)))
    return model


def predict_test_set(model):
    test_features = genfromtxt("{}/data/parsed_data/fars_test.csv".format(rootdir),
                               delimiter=",")  # We don't have labels for this one
    classification_results = model.predict(test_features)
    return classification_results


def output_results(results):
    with open("{}/result/test_set_predictions_output.csv".format(rootdir), "w+") as output:
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
    print("Done! Output will be in a csv file called 'test_set_predictions_output' in the\
'result' folder in the root directory of this project.")
