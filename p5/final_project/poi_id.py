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
import sklearn.model_selection as model_selection
import sklearn.cross_validation as cross_validation
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import fbeta_score, make_scorer, recall_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import StratifiedShuffleSplit

from tester import dump_classifier_and_data, load_classifier_and_data, test_classifier


def fit_cv(cls, features_train, labels_train, name, beta=1, folds=10):
    cls.fit(features_train, labels_train)
    cv_results = cv_scores(cls, features_train, labels_train,
                           folds=folds, beta=beta)
    return to_df(name, cv_results)


def to_df(name, scores):
    temp = pd.DataFrame()
    for key, value in scores.iteritems():
        temp[key] = value
    temp['classifier'] = name

    return temp


def cv_scores(estimator, features, labels,  beta=1, folds=20):
    skf = cross_validation.StratifiedShuffleSplit(
        labels, folds, random_state=42)
    accuracy = cross_val_score(
        estimator, features, labels, cv=skf, scoring='accuracy')
    recall = cross_val_score(
        estimator, features, labels, cv=skf, scoring='recall')
    precision = cross_val_score(
        estimator, features, labels, cv=skf, scoring='precision')
    fbeta = cross_val_score(estimator, features, labels, cv=skf,
                            scoring=make_scorer(fbeta_score, beta=beta))

    return {'f{}_score'.format(beta): fbeta,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
            }


# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".

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

cols = [col for col in df.columns if col not in [
    'name', 'poi', 'email_address']]


df['fraction_of_messages_to_poi'] = df.from_this_person_to_poi / df.from_messages
df['fraction_of_messages_from_poi'] = df.from_poi_to_this_person / df.to_messages


df[cols] = MinMaxScaler().fit_transform(df[cols])

df['total_financial_benefits'] = df.salary + df.bonus + \
    df.total_stock_value + df.exercised_stock_options


df.to_csv('cleaned_data.csv', index=False)

# All features + the created ones
selected_features = list(df.columns)
selected_features.remove('email_address')
selected_features.remove('name')
selected_features.remove('poi')
features_list = ["poi"] + selected_features
new_features_list = ['fraction_of_messages_to_poi',
                     'fraction_of_messages_from_poi', 'total_financial_benefits']
features_list = features_list + new_features_list

# Store to data_dict for easy export below.
my_dataset = df.set_index(['name']).to_dict('index')

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

kbest = SelectKBest(k='all')
kbest.fit(features, labels)
scores = zip(features_list[1:], kbest.scores_)
pd.DataFrame(scores, columns=[
             'var', 'score']).to_csv('kbest.csv', index=False)

features_list.remove('deferral_payments')
features_list.remove('restricted_stock_deferred')
features_list.remove('director_fees')
features_list.remove('to_messages')
features_list.remove('from_messages')
features_list.remove('salary')
features_list.remove('bonus')
features_list.remove('total_stock_value')
features_list.remove('exercised_stock_options')


# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html


data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)


from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)


from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

beta = 1
folds = 500

cv_train_results_df = pd.DataFrame()
# cv_train_results_df = fit_cv(GaussianNB(), features_train, labels_train,
#                              'GaussianNB', beta=beta, folds=folds)

# cv_train_results_df = cv_train_results_df.append(fit_cv(SVC(), features_train, labels_train,
#                                                         'SVC', beta=beta, folds=folds))

# cv_train_results_df = cv_train_results_df.append(fit_cv(DecisionTreeClassifier(), features_train, labels_train,
#                                                         'DecisionTreeClassifier', beta=beta, folds=folds))
# cv_train_results_df = cv_train_results_df.append(fit_cv(RandomForestClassifier(), features_train, labels_train,
#                                                         'RandomForestClassifier', beta=beta, folds=folds))

# cv_train_results_df = cv_train_results_df.append(fit_cv(LogisticRegression(C=10, tol=1), features_train, labels_train,
#                                                         'LogisticRegression', beta=beta, folds=folds))

cls = GaussianNB()
cls.fit(features_train, labels_train)
test_classifier(cls, my_dataset, features_list)

cls = SVC()
cls.fit(features_train, labels_train)
test_classifier(cls, my_dataset, features_list)

cls = DecisionTreeClassifier()
cls.fit(features_train, labels_train)
test_classifier(cls, my_dataset, features_list)

cls = LogisticRegression()
cls.fit(features_train, labels_train)
test_classifier(cls, my_dataset, features_list)


# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


pipe = Pipeline([
    ('reduce_dim', None),
    ('classify', None)
])

refcv_params = {
    'reduce_dim': [RFECV(LogisticRegression(), step=1, cv=model_selection.StratifiedKFold(2))]
}

kbest_params = {
    'reduce_dim': [SelectKBest()],
    'reduce_dim__k': list(range(1, len(features_list) - 4, 1))
}

pca_params = {
    'reduce_dim': [PCA(iterated_power=7)],
    'reduce_dim__n_components': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'reduce_dim__whiten': [False],
    'reduce_dim__copy': [True, False]
}

gaussian_nb_params = {
    'classify': [GaussianNB()]
}

svc_params = {
    'classify': [SVC()],
    'classify__kernel': ['rbf', 'linear', 'poly'],
    'classify__C': [1, 5, 10, 50, 100, 200, 500],
    'classify__gamma': list(np.arange(0.1, 0.9, 0.1))
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

classifier_params = [
    gaussian_nb_params,
    decision_tree_params,
    svc_params,
    logistic_regression_params]
reducer_params = [pca_params,  refcv_params, kbest_params]

param_grid = []

for classifier_param in classifier_params:
    for reducer_param in reducer_params:
        params = dict(reducer_param.items() + classifier_param.items())
        param_grid.append(params)


grid = GridSearchCV(pipe, param_grid=param_grid,
                    scoring=make_scorer(fbeta_score, beta=1))

grid.fit(features_train, labels_train)

clf = grid.best_estimator_


# Cross validation
cv_scores = cv_scores(clf, features_train, labels_train, beta=beta)
cv_train_results_df = cv_train_results_df.append(to_df('final', cv_scores))

cv_train_results_df.to_csv('train_results.csv', index=False)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

test_classifier(clf, my_dataset, features_list)
