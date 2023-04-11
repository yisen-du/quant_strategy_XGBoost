#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This module is used to fine-tune the models.
Given the factor data, we can fine tune the models using grid search.
The fine-tuned parameters include
    max_depth
    n_estimators
    colsample_bytree
    learning_rate
    subsample
The output is generated in the terminal
"""


import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def load_data(data_path):
    # Load the dataset
    os.chdir("../../")
    data = pd.read_csv(data_path)[['date', 'factor_indicator', 'y']]
    length = len(data)
    split_point = int(0.95 * length)

    # Split the dataset into training and testing sets
    input_train = data.iloc[0:split_point, 1:2]
    input_test = data.iloc[split_point:, 1:2]
    output_train = data.iloc[0:split_point, 2]
    output_test = data.iloc[split_point:, 2]

    # Convert the data to DMatrix format
    dtrain = xgb.DMatrix(input_train.values, label=output_train.values)
    dtest = xgb.DMatrix(input_test.values, label=output_test.values)

    return dtrain, dtest, output_train, output_test, data, input_train


def train_model(dataset_train, params):
    # Train the model using XGBoost
    num_rounds = 30
    xgb_model = xgb.train(params, dataset_train, num_boost_round=num_rounds)

    return xgb_model


def evaluate_model(trained_model, dataset_train, dataset_test, output_train, output_test):
    # Make predictions on the training and test data
    y_train_pred = trained_model.predict(dataset_train)
    y_train_pred_binary = np.where(y_train_pred >= 0.5, 1, 0)

    y_test_pred = trained_model.predict(dataset_test)
    y_test_pred_binary = np.where(y_test_pred >= 0.5, 1, 0)

    # Calculate the accuracy scores
    train_accuracy = accuracy_score(output_train, y_train_pred_binary)
    test_accuracy = accuracy_score(output_test, y_test_pred_binary)

    # Print the accuracy scores
    print('Training Accuracy: %.2f%%' % (train_accuracy * 100.0))
    print('Testing Accuracy: %.2f%%' % (test_accuracy * 100.0))


def fine_tune_model(dataset_train, output_train):
    # Define the hyperparameters to tune
    param_grid = {
        'max_depth': [x for x in range(3, 10)],
        'learning_rate': [0.01 * x for x in range(1, 30, 10)],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
        'gamma': [0.1 * x for x in range(1, 10)]
    }

    estimator = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=2,
        min_child_weight=1,
        gamma=0.1,
        reg_alpha=0.01,
        reg_lambda=0.01
    )

    # Perform grid search cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(estimator=estimator,
                               param_grid=param_grid,
                               cv=5,
                               scoring='accuracy',
                               verbose=2,
                               n_jobs=-1)
    grid_search.fit(dataset_train, output_train)

    # Print the best hyperparameters
    print("Best parameters found: ", grid_search.best_params_)

    return grid_search.best_params_


if __name__ == "__main__":
    factor_list = ['basis_spot', 'profit', 'consume_1']
    gap_day_list = [2, 7, 5]
    factor_choice = 0  # 0: basis factor, 1: profit factor, 2: consumption factor

    # Load the data
    factor = factor_list[factor_choice]
    gap_day = gap_day_list[factor_choice]
    print(factor, gap_day)
    trainSet, testSet, y_train, y_test, data, X_train = load_data(f"indicator_data/dataset_{factor}_{gap_day}.csv")

    # Fine-tune the hyperparameters using GridCV
    best_params = fine_tune_model(X_train, y_train)

    # Train the model with the best hyperparameters
    model = train_model(trainSet, best_params)
    print(model)

    # Evaluate the model
    evaluate_model(model, trainSet, testSet, y_train, y_test)
