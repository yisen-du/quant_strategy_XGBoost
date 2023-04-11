#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This module is used to generate the factors and indicators.
Given the raw data, we can compute different factors.
The output factors are saved at the path indicator_data/. They include
    basis factor with the gap days 2
    profit factor with the gap days 7
    consumption factor with the gap days 5
"""


import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def load_raw_data(factor_name):
    # return factor data and price data
    df_future_price = pd.read_csv("raw_data/CF_price.csv")
    df_factor = pd.read_csv(f"raw_data/CF_{factor_name}.csv")
    df_merge = df_future_price.merge(df_factor,
                                     left_on="CLOCK",
                                     right_on="SymbolDate")[['CLOCK', 'CLOSE', 'SymbolVal']]
    df_merge.columns = ['date', 'futurePrice', 'factor']
    return df_merge


def construct_signal(df, factor_name, gap_day):
    # input
    # for basis: need to compute the factor value
    if factor_name == 'basis_spot':
        df['factor'] = df['factor'] - df['futurePrice']

    # shift 1 day to avoid using future information
    df['factor_shift'] = df['factor'].shift(1)
    df_input = df[['date', 'factor_shift']].dropna()
    df_input.columns = ['date', 'factor_indicator']

    # output
    # outcome variable is the future price direction
    df_output = df[['date', 'futurePrice']]
    df_output['gap_days_later_close'] = df['futurePrice'].shift(-gap_day)
    df_output = df_output.dropna()
    # here shift again to avoid using the close price of today
    df_output['y'] = np.where((df_output['gap_days_later_close'] - df_output['futurePrice'].shift(1)) >= 0, 1, 0)
    df_output = df_output[['date', 'y']]
    df_output = df_output.dropna()

    # merge input and output
    df_dataset = df_input.merge(df_output, on='date')
    df_dataset = df_dataset[['date', 'factor_indicator', 'y']]
    return df_dataset


def construct_signal_multi_factor(df, gap_day):
    df['basis_factor'] = df['basis_factor'] - df['futurePrice']

    # shift 1 day to avoid using future information
    df['basis_factor_shift'] = df['basis_factor'].shift(1)
    df['profit_factor_shift'] = df['profit_factor'].shift(1)
    df['consumption_factor_shift'] = df['consumption_factor'].shift(1)
    df_input = df[['date', 'basis_factor_shift', 'profit_factor_shift', 'consumption_factor_shift']].dropna()

    # output
    # outcome variable is the future price direction
    df_output = df[['date', 'futurePrice']]
    df_output['gap_days_later_close'] = df['futurePrice'].shift(-gap_day)
    df_output = df_output.dropna()
    df_output['y'] = np.where((df_output['gap_days_later_close'] - df_output['futurePrice'].shift(1)) >= 0, 1, 0)
    df_output = df_output[['date', 'y']]

    # merge input and output
    df_dataset = df_input.merge(df_output, on='date')
    df_dataset = df_dataset[['date', 'basis_factor_shift', 'profit_factor_shift', 'consumption_factor_shift', 'y']]
    return df_dataset


if __name__ == "__main__":
    os.chdir("../../")
    factor_list = ['basis_spot', 'profit', 'consume_1']
    gap_day_list = [2, 7, 5]

    # construct indicator_data for simple factor
    for i in range(len(factor_list)):
        factor = factor_list[i]
        gap_day = gap_day_list[i]
        print(factor, gap_day)
        data_merge = load_raw_data(factor)
        dataset = construct_signal(data_merge, factor_name=factor, gap_day=gap_day)
        dataset.to_csv(f"indicator_data/dataset_{factor}_{gap_day}.csv")
        print(f"save as indicator_data/dataset_{factor}_{gap_day}.csv")

    # construct indicator_data for multiple factors
    gap_day = 5
    data_part_1 = load_raw_data(factor_list[0])
    data_part_2 = load_raw_data(factor_list[1])
    data_part_3 = load_raw_data(factor_list[2])
    data_merge = data_part_1[['date', 'factor']].merge(data_part_2[['date', 'factor']], on="date")
    data_merge = \
        data_merge[['date', 'factor_x', 'factor_y']].merge(data_part_3[['date', 'factor', 'futurePrice']], on="date")
    data_merge.columns = ['date', 'basis_factor', 'profit_factor', 'consumption_factor', 'futurePrice']

    dataset = construct_signal_multi_factor(data_merge, gap_day=gap_day)
    dataset.to_csv(f"indicator_data/dataset_multi_factor_{gap_day}.csv")
    print(f"save as indicator_data/dataset_multi_factor_{gap_day}.csv")
