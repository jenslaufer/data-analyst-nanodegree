#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.cross_validation import train_test_split
from time import time
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score
from sklearn.model_selection import GridSearchCV


class ClassifierTuner:

    def __init__(self, classifier, features, labels, test_size,
                 test_split_random_state, beta=1., param_grid=[{}]):
        features_train, features_test, labels_train, labels_test = train_test_split(
            features, labels, test_size=test_size, random_state=test_split_random_state)
        self.classifier_ = classifier
        self.__features_train = features_train
        self.__labels_train = labels_train
        self.__features_test = features_test
        self.__labels_test = labels_test
        self.__beta = beta
        self.__name = type(classifier).__name__
        self.__param_grid = param_grid

        self.__tuning()

    def __tuning(self):
        print "tuning..."
        grid_search = GridSearchCV(self.classifier_, self.__param_grid)
        grid_search.fit(self.__features_train, self.__labels_train)

        self.tuning_results_ = grid_search.cv_results_
        self.best_score_ = grid_search.best_score_
        self.classifier_ = grid_search.best_estimator_ 

        print "tuned."

    def metrics(self):
        metrics_ = {}
        predicted = self.classifier_.predict(self.__features_test)

        metrics_['beta'] = self.__beta
        metrics_['accuracy_score'] = round(accuracy_score(
            self.__labels_test, predicted), 3)
        metrics_['recall_score'] = round(recall_score(
            self.__labels_test, predicted), 3)
        metrics_['precision_score'] = round(precision_score(
            self.__labels_test, predicted), 3)
        metrics_['fbeta_score'] = round(fbeta_score(
            self.__labels_test, predicted, self.__beta), 3)

        return metrics_


# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi', 'salary']  # You will need to use more features

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


clf = ClassifierTuner(GaussianNB(), features, labels, 0.3, 42)
print clf.metrics()

clf = ClassifierTuner(SVC(), features, labels, 0.3, 42)
print clf.metrics()


# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf.classifier_, my_dataset, features_list)
