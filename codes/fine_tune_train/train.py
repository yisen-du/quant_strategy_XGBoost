#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This module is used to train the xgboost models using the fine-tuned parameters.
Given the best parameters and the factor data, we can train the model and generate the trading signals.
The output signals are saved at the path signal_data/. They include
    signals of the basis factor with the gap days 2
    signals of the profit factor with the gap days 7
    signals of the consumption factor with the gap days 5
"""


import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score


def load_data(data_path):
    # Load the dataset
    os.chdir("../../")
    if "multi_factor" in data_path:
        data = pd.read_csv(data_path)[['date',
                                       'basis_factor_shift',
                                       'profit_factor_shift',
                                       'consumption_factor_shift',
                                       'y']]
    else:
        data = pd.read_csv(data_path)[['date', 'factor_indicator', 'y']]

    length = len(data)
    split_point = int(0.95 * length)
    # Split the dataset into training and testing sets
    input_train = data.iloc[0:split_point, 1:-1]
    input_test = data.iloc[split_point:, 1:-1]
    output_train = data.iloc[0:split_point, -1]
    output_test = data.iloc[split_point:, -1]
    # Convert the data to DMatrix format
    dtrain = xgb.DMatrix(input_train.values, label=output_train.values)
    dtest = xgb.DMatrix(input_test.values, label=output_test.values)
    print(data.iloc[0:split_point, 0])
    print(data.iloc[split_point:, 0])
    print(len(data.iloc[split_point:, 0]))
    return dtrain, dtest, output_train, output_test, data


def train_model(dataset_train, best_parameters):
    # Train the model using XGBoost
    num_rounds = 30
    xgb_model = xgb.train(best_parameters, dataset_train, num_boost_round=num_rounds)

    return xgb_model


def evaluate_model(trained_model, dataset_train, dataset_test, output_train, output_test):
    # Make predictions on the training and test data
    y_train_pred = trained_model.predict(dataset_train)
    y_test_pred = trained_model.predict(dataset_test)

    # Calculate the accuracy scores
    train_accuracy = accuracy_score(output_train, y_train_pred)
    test_accuracy = accuracy_score(output_test, y_test_pred)

    # Print the accuracy scores
    print('Training Accuracy: %.2f%%' % (train_accuracy * 100.0))
    print('Testing Accuracy: %.2f%%' % (test_accuracy * 100.0))

    return y_train_pred, y_test_pred


if __name__ == "__main__":
    factor_list = ['basis_spot', 'profit', 'consume_1', 'multi_factor']
    gap_day_list = [2, 7, 5, 5]
    factor_choice = 0  # 0: basis factor, 1: profit factor, 2: consumption factor, 3: multiple factors

    # Load the data
    factor = factor_list[factor_choice]
    gap_day = gap_day_list[factor_choice]
    print(factor, gap_day)
    trainSet, testSet, y_train, y_test, data = load_data(f"indicator_data/dataset_{factor}_{gap_day}.csv")

    best_para_list = [
        {
            'max_depth': 7,
            'objective': "multi:softmax",
            'num_class': 2,
            'learning_rate': 0.3,
            'subsample': 0.8},
        {
            'max_depth': 8,
            'objective': "multi:softmax",
            'num_class': 2,
            'learning_rate': 0.3,
            'subsample': 0.6},
        {
            'max_depth': 9,
            'objective': "multi:softmax",
            'num_class': 2,
            'learning_rate': 0.3,
            'subsample': 0.6},
        {
            'max_depth': 4,
            'objective': "multi:softmax",
            'num_class': 2,
            'learning_rate': 0.3,
            'colsample_bytree': 0.9,
            'subsample': 0.6}
    ]

    # Train the model
    model = train_model(trainSet, best_parameters=best_para_list[factor_choice])

    # Evaluate the model
    train_signal, test_signal = evaluate_model(model, trainSet, testSet, y_train, y_test)

    # Save the signals
    data['signal'] = train_signal.tolist() + test_signal.tolist()
    data.to_csv(f"signal_data/signal_{factor}_{gap_day}.csv")
    print(f"The signal file is saved as signal_data/signal_{factor}_{gap_day}.csv")
