"""
Trains a RandomForestClassifier to recognise real and fake stars from a training and test dataset.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

training_data = pd.read_csv("data/training_data.csv")
training_data = training_data.drop(columns=["HIP"])

training_data_X = training_data[["RAICRS", "DEICRS", "Vmag"]]
training_data_Y = training_data["label"]

clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=0)
clf.fit(training_data_X, training_data_Y)

test_data = pd.read_csv("data/test_data.csv")
test_data = test_data.drop(columns=["HIP"])

test_data_X = test_data[["RAICRS", "DEICRS", "Vmag"]]
test_data_Y = test_data["label"]

test_results = clf.predict(test_data_X)
print(classification_report(test_data_Y, test_results))