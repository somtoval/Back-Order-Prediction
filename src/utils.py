import os
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.logger import logging

import pickle

from src.exception import CustomException

# Defining a function that takes in 2 arguments (file_path of object to be store and object to be store)
def save_object(file_path, obj):
    try:
        # extracts the file directory name from the file path
        dir_path = os.path.dirname(file_path)

        # makes a directory using the extracted directory name
        os.makedirs(dir_path, exist_ok=True)

        # opens the created file path and dumps the obj using pickle.dump()
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Raising an exception based on our custom exception class we have created and imported, it takes in the exception and sys( to get more info about  it allows the exception to include information about the system state or the Python interpreter)
        raise CustomException(e, sys)

# Defining a function to load object that takes in file path as input argument    
def load_object(file_path):
    try:
        # Opening your the file path passed
        with open(file_path, 'rb') as file_obj:
            # Returning the loaded file after using pickle to load it
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occured in load_object function utils')
        raise CustomException(e, sys)

# This function takes in input values and target values and the models for evaluation which is in a dictionary
def evaluate_models(X, y, models):
    try:
        # Splitting the X data and Y data into X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialization a dictionary that will hold the model report
        report = {}

        # Looping through the models dictionary passed in, note: when we do list(a_dictionary) it gives a python list made up of the dictionary key
        for i in range(len(list(models))):
            # Assigning the ith model object to a list to be called model
            model = list(models.values())[i]
            # Fitting the model to the train data
            model.fit(X_train, y_train)
            # Predicting the train data using the model
            y_train_pred = model.predict(X_train)
            # Predicting the test data using the model
            y_test_pred = model.predict(X_test)
            # Computing the accuracy score of our train data
            train_model_score = accuracy_score(y_train, y_train_pred)
            # Computing the accuracy score of our test data
            test_model_score = accuracy_score(y_test, y_test_pred)
            # Adding the model used as key and it's test score as value to our report dictionary
            report[list(models.keys())[i]] = test_model_score
        # This function returns the report dictionary
        return report

    except Exception as e:
        raise CustomException(e, sys)
