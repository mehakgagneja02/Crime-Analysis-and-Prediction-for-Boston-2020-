"""
Copyright (c) 2022
Written by : Mehak Gagneja
Description: Prediction Models
"""

# import the required library: here I would need panda package for reading the files
import pandas as pd
import numpy as np

# Libraries for Prediction Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
    
def k_neighbors_model(df):
    # Add training and testing to the model.
    # 75% of the dataset is utilized in training the model
    # 25% is used as a test data
    # X_train and X_test: stores the Independent variables
    # y_train, y_test: stores the Dependent variables
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('SHOOTING', axis = 1),
        df.SHOOTING, test_size=0.25,
        random_state=42
    )
    # Declare the KNeighborsClassifier that will use 5 neighbors to predict the value
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train)
    # Test the model
    model_accuracy = neigh.score(X_test, y_test)

    print(f'This model is {model_accuracy} Accurate')
    
    # Predict results
    y_pred = neigh.predict(X_test)
    print(f'\n Total shooting predicted are : {np.sum(y_pred)}')
    print(f'\n Actual shooting Involved are : {np.sum(y_test)}') 
    confusion_matrix(y_test, y_pred)
    return y_pred;
    
def decisiontree_model(df):
    # Add training and testing to the model.
    # 75% of the dataset is utilized in training the model
    # 25% is used as a test data
    # X_train and X_test: stores the Independent variables
    # y_train, y_test: stores the Dependent variables
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('SHOOTING', axis = 1),
        df.SHOOTING, test_size=0.25,
        random_state=42
    )
    clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
    clf_gini.fit(X_train, y_train)
    plt.figure(figsize=(18,12))
    tree.plot_tree(clf_gini, feature_names=X_train.columns, fontsize=30)
    #Predict the values 
    y_pred_gini = clf_gini.predict(X_test)
    print('\n Decision Trees classification Model accuracy score with criterion gini index: {0:0.4f}'.format(accuracy_score(y_test, y_pred_gini)))
    return y_pred_gini;