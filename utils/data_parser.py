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

NUMERIC_CLASSES = ['AGE', 'ALCOHOL_TEST_RESULT', 'DRUG_TEST_RESULTS_(1_of_3)', 'DRUG_TEST_RESULTS_(2_of_3)', 'DRUG_TEST_RESULTS_(3_of_3)']

def parse_headers():
  """
  We use indices instead of a map because we lack column names on the datasets.
  """
  os.chdir("../data/raw_data/")
  file = open("headers.txt", "r")
  dataset_features = []
  for line in file:
    header, values = line.split()

    if header not in NUMERIC_CLASSES: 
      values_list = (re.sub(r"{|}", "", values)).split(",")
      values_map = {}
      index = 1
      for value in values_list:
        values_map[value] = index
        index += 1
      dataset_features.append(values_map)
    else:
      dataset_features.append({header: -1})

  print(dataset_features)

if __name__ == "__main__":
  parse_headers()