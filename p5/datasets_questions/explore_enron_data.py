#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle
import pandas as pd
import pprint

pp = pprint.PrettyPrinter(indent=4)

enron_data = pickle.load(
    open("../final_project/final_project_dataset.pkl", "r"))


pp.pprint(enron_data['PRENTICE JAMES']['total_stock_value'])
pp.pprint(enron_data['COLWELL WESLEY']['from_this_person_to_poi'])
pp.pprint(enron_data['SKILLING JEFFREY K']['exercised_stock_options'])


def pois(data):
    pois = {}
    for key, person in data.iteritems():
        if person['poi'] == True:
            pois[key] = person
    return pois


def nan(data, feature):
    num_nan = 0
    num_n_nan = 0

    for key, person in data.iteritems():
        if person[feature] != 'NaN':
            num_n_nan += 1
        else:
            num_nan += 1
    return (num_nan, num_n_nan, num_nan + num_n_nan, round(float(num_n_nan) / float(num_nan + num_n_nan) * 100, 2))


for feature in list(enron_data['PRENTICE JAMES'].keys()):
    num_nan, num_n_nan, total, pct_n_nan = nan(enron_data, feature)
    pp.pprint("{}: n_nan:{}, nan:{}, total:{} -> pct_n_nan:{}".format(feature, num_n_nan, num_nan, total, pct_n_nan))

pp.pprint("------")
for feature in list(enron_data['PRENTICE JAMES'].keys()):
    num_nan, num_n_nan, total, pct_n_nan = nan(pois(enron_data), feature)
    pp.pprint("{}: n_nan:{}, nan:{}, total:{} -> pct_n_nan:{}".format(feature, num_n_nan, num_nan, total, pct_n_nan))
