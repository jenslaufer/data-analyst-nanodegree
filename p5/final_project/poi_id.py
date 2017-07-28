#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.cross_validation import train_test_split
from time import time
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer, recall_score, accuracy_score
from sklearn.model_selection import cross_val_score


import pandas as pd
import seaborn as sb
import math
import matplotlib.pyplot as plt


class ClassifierTuner:

    def __init__(self, classifier, features, labels, test_size,
                 test_split_random_state, beta=1, param_grid=[{}], cv=3):
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
        self.__cv = cv
        self.__beta_scorer = make_scorer(fbeta_score, beta=self.__beta)

        self.__tuning()

    def __tuning(self):
        print "tuning {}...".format(self.__name)

        grid_search = GridSearchCV(
            self.classifier_, self.__param_grid, scoring=self.__beta_scorer, cv=self.__cv)
        grid_search.fit(self.__features_train, self.__labels_train)

        self.tuning_report_ = grid_search.cv_results_
        self.classifier_ = grid_search.best_estimator_
        self.params_ = grid_search.best_params_
        self.index_ = grid_search.best_index_
        self.metrics_ = self.__metrics()

        print "tuned."

    def __metrics(self):
        metrics_ = {}
        predicted_test = self.classifier_.predict(self.__features_test)

        metrics_['name'] = self.__name
        metrics_['test_accuracy_score'] = round(accuracy_score(
            self.__labels_test, predicted_test), 3)
        metrics_['test_recall_score'] = round(recall_score(
            self.__labels_test, predicted_test), 3)
        metrics_['test_precision_score'] = round(precision_score(
            self.__labels_test, predicted_test), 3)
        metrics_['test_f{}_score'.format(self.__beta)] = round(fbeta_score(
            self.__labels_test, predicted_test, self.__beta), 3)

        vals = cross_val_score(
            self.classifier_, self.__features_train, self.__labels_train, scoring='accuracy', cv=self.__cv)
        metrics_['cv_train_accuracy_score_mean'] = round(vals.mean(), 3)
        metrics_['cv_train_accuracy_score_std'] = round(vals.std(), 3)

        vals = cross_val_score(
            self.classifier_, self.__features_train, self.__labels_train, scoring='precision', cv=self.__cv)
        metrics_['cv_train_precision_score_mean'] = round(vals.mean(), 3)
        metrics_['cv_train_precision_score_std'] = round(vals.std(), 3)

        vals = cross_val_score(
            self.classifier_, self.__features_train, self.__labels_train, scoring='recall', cv=self.__cv)
        metrics_['cv_train_recall_score_mean'] = round(vals.mean(), 3)
        metrics_['cv_train_recall_score_std'] = round(vals.std(), 3)

        vals = cross_val_score(
            self.classifier_, self.__features_train, self.__labels_train, scoring=self.__beta_scorer, cv=self.__cv)
        metrics_['cv_f{}_beta_mean'.format(
            self.__beta)] = round(vals.mean(), 3)
        metrics_['cv_f{}_beta_std'.format(
            self.__beta)] = round(vals.std(), 3)

        return metrics_

class Plot:
    def __init__(self, df):
        self.df_ = df

    def boxplot(self, column, ax=None):
        sb.boxplot(data=self.df_, x='poi', y=column, ax=ax)

        head = self.df_.sort_values(by=[column], ascending=[False]).head(10)
        tail = self.df_.sort_values(by=[column], ascending=[False]).tail(10)

        def ann(row):
            ind = row[0]
            r = row[1]
            plt.annotate(r['name'], xy=(r["poi"], r[column]),
                         xytext=(2, 2), textcoords="offset points")

        for row in head.iterrows():
            ann(row)
        for row in tail.iterrows():
            ann(row)
        if ax == None:
            plt.show()


    def boxplots(self):
        for feature in list(self.df_.columns):
            fig, ax = plt.subplots(1, 1)
            if feature != 'name' and feature != 'email_address':
                self.boxplot(feature, ax)
        plt.show()


    def boxplot_grid(self):

        features = list(self.df_.columns)
        features.remove('name')
        features.remove('email_address')
        features.remove('poi')

        height = int(math.ceil(len(features) / 2))

        fig = plt.figure()
        fig.set_figheight(15 * height)
        fig.set_figwidth(15)

        axes = []
        for i in range(0, height):
            for j in range(0, 2):
                axes.append(plt.subplot2grid((height, 2), (i, j)))

        for feature in features:
            boxplot(self.df_, feature, ax=axes[features.index(feature)])

        plt.show()

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi', 'salary']  # You will need to use more features

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


data = []
for key, value in data_dict.iteritems():
    value['name'] = key
    data.append(value)

df = pd.DataFrame(data)
df = df.replace({'NaN': 0})

df = df[(df.name != 'TOTAL') & (df.name != 'THE TRAVEL AGENCY IN THE PARK')]

plot = Plot(df)
plot.boxplots()


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


clf = ClassifierTuner(GaussianNB(), features, labels, 0.3, 42, cv=20)
print clf.metrics_
print clf.params_
print clf.tuning_report_


# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf.classifier_, my_dataset, features_list)
