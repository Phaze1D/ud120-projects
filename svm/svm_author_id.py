#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
import pandas as pd
from time import time
sys.path.append("../tools/")
sys.path.append("../choose_your_own/")
from email_preprocess import preprocess
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


clf = SVC(C=10000.0, kernel='rbf')

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"


t1 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t1, 3), "s"


print accuracy_score(pred, labels_test)

# df = pd.DataFrame({'pred':pred})
# print len(df[df.pred == 1])

# try:
#     prettyPicture(clf, features_test, labels_test)
# except NameError:
#     pass
