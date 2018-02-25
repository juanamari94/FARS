import csv
import os
import re

INJURY_CLASSES = {'Possible_Injury': 0,
                  'No_Injury': 1,
                  'Incapaciting_Injury': 6,
                  'Fatal_Injury': 3,
                  'Unknown': 4,
                  'Nonincapaciting_Evident_Injury': 5,
                  'Died_Prior_to_Accident': 2,
                  'Injured_Severity_Unknown': 7}

NUMERIC_CLASSES = ['AGE', 'ALCOHOL_TEST_RESULT', 'DRUG_TEST_RESULTS_(1_of_3)', 'DRUG_TEST_RESULTS_(2_of_3)',
                   'DRUG_TEST_RESULTS_(3_of_3)']


def create_header_mappings():
    """
    We use indices instead of a map because we lack column names on the datasets.
    """
    file = open("headers.txt", "r")
    dataset_features_and_labels = []
    for line in file:
        header, values = line.split()

        if header not in NUMERIC_CLASSES:
            values_list = (re.sub(r"[{}]", "", values)).split(",")
            values_map = {}
            index = 1
            for value in values_list:
                values_map[value] = index
                index += 1
            dataset_features_and_labels.append(values_map)
        else:
            dataset_features_and_labels.append(-1)

    dataset_features_and_labels.append(INJURY_CLASSES)

    return dataset_features_and_labels


def parse_dataset(filename, mappings):
    with open(filename, "r") as train_set:
        train_set_reader = csv.reader(train_set, delimiter=",")
        with open("../parsed_data/{}".format(filename), "w+") as parsed_train_set:
            train_set_writer = csv.writer(parsed_train_set, delimiter=",")
            for row in train_set_reader:
                mapped_row = []
                for i in range(0, len(row)):
                    if mappings[i] != -1:
                        mapped_row.append(mappings[i][row[i]])
                    else:
                        mapped_row.append(row[i])
                train_set_writer.writerow(mapped_row)


if __name__ == "__main__":
    os.chdir("../data/raw_data/")
    header_mappings = create_header_mappings()
    parse_dataset("fars_train.csv", header_mappings)
    parse_dataset("fars_test.csv", header_mappings)
