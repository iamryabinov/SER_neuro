from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from datasets.iemocap import *
from datasets.iemocap_egemaps import *

import pandas as pd
import sys
import numpy as np
import pickle
import time

def my_train_test_split(dataset):
    X = dataset.features.X
    y = dataset.features.y
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

# Implement a grid search over parameters of a nonlinear rbf SVM.
svm_parameters = [{
                     'kernel': ['rbf'],
                     'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100],
                 }]
svm = SVC()
grid = GridSearchCV(svm, param_grid=svm_parameters, n_jobs=-1, cv=5, verbose=1)

def svm_without_pca():
    X_train, X_test, y_train, y_test = my_train_test_split(iemocap_four_labels)
    print(len(X_train) + len(X_test))
    grid.fit(X_train, y_train)
    best_svm = grid.best_estimator_
    y_pred = best_svm.predict(X_test)
    report_dict = classification_report(y_test, y_pred, digits=2, output_dict=True)
    print('==============================')
    print(grid.best_estimator_)
    print(report_dict)


def svm_with_pca():
    X = iemocap_four_labels.features.X
    y = iemocap_four_labels.features.y
    pca = PCA(n_components=0.95, svd_solver='full')
    X = np.ascontiguousarray(pca.fit_transform(X))
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
    grid.fit(X_train, y_train)
    best_svm = grid.best_estimator_
    y_pred = best_svm.predict(X_test)
    report_dict = classification_report(y_test, y_pred, digits=2, output_dict=True)
    print('==============================')
    print(grid.best_estimator_)
    print(report_dict)



if __name__ == '__main__':
    svm_without_pca()
    svm_with_pca()