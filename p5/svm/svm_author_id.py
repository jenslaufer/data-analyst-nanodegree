#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
import numpy as np
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
from sklearn.svm import SVC


clf = SVC(kernel='rbf', C=10000)
clf.fit(features_train,
        labels_train)
print clf.predict(list(features_test[i] for i in [10, 26, 50]))
preds = clf.predict(features_test)

count = 0
for pred in preds:
    if pred == 1:
        count += 1
print count

print len(filter(lambda x: x == 1, preds))


cs = [10, 100, 1000, 10000]
for c in cs:
    clf = SVC(kernel='rbf', C=c)
    clf.fit(features_train[:len(features_train) / 100],
            labels_train[:len(labels_train) / 100])
    print "c={}:{}".format(c, clf.score(features_test, labels_test))


clf = SVC(kernel='rbf', C=10000)
clf.fit(features_train,
        labels_train)
print "c={}:{}".format(c, clf.score(features_test, labels_test))
#########################################################
