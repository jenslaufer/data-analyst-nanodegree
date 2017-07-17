#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


# read in data dictionary, convert to numpy array
data_dict = pickle.load(
    open("../final_project/final_project_dataset.pkl", "r"))
del data_dict['TOTAL']
data_dict
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


# your code below

target, features = targetFeatureSplit(data)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
fit = reg.fit(features, target)


print "bonus = {} * salary + ({})".format(reg.coef_[0], reg.intercept_)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)
    matplotlib.pyplot.annotate(str(salary)+","+str(bonus), xy=(salary, bonus))

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()