#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print features_train.shape

gnb = GaussianNB()

t0 = time()
gnb.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
gpredict = gnb.predict(features_test)
print "training time:", round(time()-t1, 3), "s"


print accuracy_score(gpredict, labels_test)


#
# mnb = MultinomialNB()
# mnb.fit(features_train, labels_train)
# mpredict = mnb.predict(features_test)
# print accuracy_score(mpredict, labels_test)
#
# bnb = BernoulliNB()
# bnb.fit(features_train, labels_train)
# bpredict = bnb.predict(features_test)
# print accuracy_score(bpredict, labels_test)



#########################################################
### your code goes here ###


#########################################################
