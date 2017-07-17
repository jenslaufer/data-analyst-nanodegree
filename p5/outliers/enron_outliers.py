#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


# read in data dictionary, convert to numpy array
data_dict = pickle.load(
    open("../final_project/final_project_dataset.pkl", "r"))
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


# your code below

target, features = targetFeatureSplit(data)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
fit = reg.fit(features, target)


print "bonus = {} * salary + ({})".format(reg.coef_[0], reg.intercept_)