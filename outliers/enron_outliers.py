#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
import numpy as np

# Clean data - remove single outlier
idx_max_bonus = data[:,1].argsort()[-1]
# data = data[data[:,1].argsort()][:-1]
data_cleaned = np.delete(data, idx_max_bonus, axis=0)

data_max_bonus = [(k,v['bonus']) for k,v in data_dict.iteritems() if v['bonus']==data[idx_max_bonus][1]]
print 'Label for outlier (max bonus): {0}'.format(data_max_bonus[0][0])

# View cleaned data
# data = data_cleaned


# Alternatively, remove outlier entry from data_dict, then plot
data_dict.pop(data_max_bonus[0][0])
data = featureFormat(data_dict, features)


# Plot
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()