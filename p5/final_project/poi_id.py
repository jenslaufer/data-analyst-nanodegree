#!/usr/bin/python
import sys

import pickle

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import fbeta_score, make_scorer, recall_score, accuracy_score
from sklearn.feature_selection import chi2

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

from tester import dump_classifier_and_data, load_classifier_and_data, test_classifier

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees',
                 'exercised_stock_options', 'expenses',
                 'from_messages', 'from_poi_to_this_person',
                 'from_this_person_to_poi', 'loan_advances', 'long_term_incentive',
                 'other', 'restricted_stock',
                 'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi',
                 'to_messages', 'total_payments', 'total_stock_value']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

flattened = []
for key, value in data_dict.iteritems():
    value['name'] = key
    flattened.append(value)

df = pd.DataFrame.from_dict(flattened)

df.to_csv('raw_data.csv', index=False)

# cleaning up

df = df.replace('NaN', np.nan)

df['email_address'] = df.email_address.fillna('')
df = df.fillna(0)

# Task 2: Remove outliers
df = df[df.name != 'TOTAL']
df = df[df.name != 'THE TRAVEL AGENCY IN THE PARK']
df = df[df.name != 'LOCKHART EUGENE E']


# Task 3: Create new feature(s)
df = df.replace({'from_messages': {0: df.from_messages.mean()}})
df = df.replace({'to_messages': {0: df.to_messages.mean()}})


df['fraction_of_messages_to_poi'] = df.from_this_person_to_poi / df.from_messages
df['fraction_of_messages_from_poi'] = df.from_poi_to_this_person / df.to_messages
df['total_financial_benefits'] = df.salary + df.bonus + \
    df.total_stock_value + df.exercised_stock_options

df.to_csv('cleaned_data.csv', index=False)

# Store to data_dict for easy export below.

my_dataset = df.set_index(['name']).to_dict('index')

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# features = MinMaxScaler().fit_transform(features)


# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


pipe = Pipeline([
    ('scale', None),
    ('reduce_dim', None),
    ('classify', None)
])


scale_params = {'scale': [MinMaxScaler()]
                }

pca_params = {
    'reduce_dim': [PCA(iterated_power=7)],
    'reduce_dim__n_components': [0.3, 0.4, 0.5, 0.6, 0.7],
    'reduce_dim__whiten': [False],
    'reduce_dim__copy': [True, False]
}

k_best_params = {
    'reduce_dim': [SelectKBest()],
    'reduce_dim__k': list(range(2, 10, 1))}

gaussian_nb_params = {
    'classify': [GaussianNB()]
}

svc_params = {
    'classify': [SVC()],
    'classify__kernel': ['rbf', 'linear', 'poly'],
    'classify__C': [1, 5, 10, 50, 100, 200, 500],
    'classify__gamma': list(np.arange(0.1, 0.9, 0.1))
}

random_forest_params = {
    'classify': [RandomForestClassifier()],
    'classify__criterion': ['gini', 'entropy'],
    'classify__min_samples_split': [10, 15, 20, 25]
}

decision_tree_params = {
    'classify': [DecisionTreeClassifier()],
    'classify__criterion': ['gini', 'entropy'],
    'classify__min_samples_split': [10, 15, 20, 25]
}

logistic_regression_params = {
    'classify': [LogisticRegression()],
    "classify__C": [0.05, 0.5, 1, 10, 10**2, 10**3, 10**5, 10**10, 10**15],
    "classify__tol": [10**-1, 10**-2, 10**-4, 10**-5, 10**-6, 10**-10, 10**-15],
    "classify__class_weight": ['balanced']
}


classifier_params = [gaussian_nb_params,
                     svc_params,
                     decision_tree_params,
                     random_forest_params,
                     logistic_regression_params]
reducer_params = [pca_params, k_best_params]

param_grid = []

for classifier_param in classifier_params:
    for reducer_param in reducer_params:
        params = dict(scale_params.items() +
                      reducer_param.items() + classifier_param.items())
        param_grid.append(params)


grid = GridSearchCV(pipe, param_grid=param_grid,
                    scoring=make_scorer(fbeta_score, beta=1))


from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

grid.fit(features_train, labels_train)

clf = grid.best_estimator_

# Cross validation


# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
