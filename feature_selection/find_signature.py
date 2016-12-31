#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
from sklearn import tree
from sklearn.metrics import accuracy_score
from time import time

clf = tree.DecisionTreeClassifier()#min_samples_split=40)

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


## Find features importance
importance_array = clf.feature_importances_

min_importance = 0.2

max_importance = max(importance_array)
index_max_importance = importance_array.argmax()

print 'Max importance: {0}. Index: {1}'.format(max_importance, index_max_importance)

# Arbitrarily viewing max 10 values
# outliers = sorted(importance_array)[-10:]
# index_of_outliers = importance_array.argsort()[-10:]

# View outliers where min_importance used
# outliers = sorted(importance_array[importance_array > min_importance])

# View word at index 24321
# vectorizer.get_feature_names()[24321]



