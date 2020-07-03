# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 20:16:17 2020

@author: Tamim
"""
import pandas as pd
import numpy as np
import pickle

## Read the data file
df=pd.read_csv('BankNote_Authentication.csv')


# Model Building
from sklearn.model_selection import train_test_split
X = df.drop(columns='class')
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

## Prediction
y_pred=classifier.predict(X_test)

### Check Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)

# Creating a pickle file for the classifier
filename = 'Bank_note-Authentication-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))