#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

# Select one of the following classifers. Comment out the rest.
clf = KNeighborsClassifier(n_neighbors=4) # k-Nearest Neighbors
# clf = AdaBoostClassifier() # AdaBoost Classifier
# clf = RandomForestClassifier() # Random Forest Classifier

# Fit the data
t0 = time()
clf = clf.fit(features_train, labels_train)
print 'Training time: {0} s'.format(round(time()-t0, 3))

# Predict
t1 = time()
labels_pred = clf.predict(features_test)
print 'Prediction time: {0} s'.format(round(time()-t1, 3))

# Check accuracy
accuracy = accuracy_score(labels_test, labels_pred)
print 'Accuracy: {0}'.format(round(accuracy, 4))

# Plot
try:
    prettyPicture(clf, features_test, labels_test)
    plt.show()
except NameError:
	print 'prettyPicture passed'
	pass
