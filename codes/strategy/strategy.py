#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This module is used to construct the strategy and do the back-testing.
Given the signal data, we can compute the return on both train set and test set.
The output results are saved at the path result/. They include
    plots of net values
    plots of position plot
    statistics of return (daily return, Sharper, ...etc)
"""


import os
import pandas as pd
import numpy as np
import datetime
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-paper')


def load_data(path):
    # read price data and signal data
    # load the signal
    df_signal = pd.read_csv(path)
    first_date = df_signal['date'].iloc[0]

    # load the price data
    df_future_price = pd.read_csv("raw_data/CF_price.csv")
    df_future_price = df_future_price[df_future_price['CLOCK'] >= first_date]
    return df_signal, df_future_price


def compute_position(df_signal, df_future_price, para_factor):
    # compute action and position based on signal
    split_point = int(len(df_signal) * 0.95)
    signal = df_signal[['date', 'signal']]
    df = df_future_price[['CLOCK', 'CLOSE']].merge(signal, left_on="CLOCK", right_on="date").drop(columns="date")
    df.columns = ['date', 'close_price', 'signal']
    df['position'] = df['signal'].diff().fillna(0)

    # if there's consecutive long and short, filter it out (optional)
    if para_factor == "basis_spot":
        df['position_filter'] = df['position'].diff().fillna(0)

        index_list_1 = df[df['position_filter'] == -2].index.values - 1
        index_list_2 = df[df['position_filter'] == 2].index.values - 1

        df['signal_filter'] = df['signal']

        df.loc[index_list_1.tolist(), 'signal_filter'] = 0
        df.loc[index_list_2.tolist(), 'signal_filter'] = 1

        df['position_2'] = df['signal_filter'].diff().fillna(0)
        df.loc[0, 'position_2'] = -1

        df_result = df[['date', 'close_price', 'signal_filter', 'position_2']]
    else:
        df_result = df[['date', 'close_price', 'signal', 'position']]

    # result
    df_result.columns = ['date', 'close_price', 'signal', 'action']
    df_result['position'] = df_result['action'].replace(0, np.nan).fillna(method='ffill').fillna(0)

    df_result_train = df_result.iloc[0:split_point]
    df_result_test = df_result.iloc[split_point:]

    return df_result_train, df_result_test


def compute_return_statistics(df):
    # read a dataframe with columns date, close_price, signal, action, and position
    # years and time
    start_date = datetime.datetime.strptime(df['date'].iloc[0], "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(df['date'].iloc[-1], "%Y-%m-%d").date()
    time_delta_years = (end_date - start_date).days / 365
    transaction_time = len(df[df['action'] != 0])
    transaction_time_average = transaction_time / time_delta_years

    # daily return
    df['daily_return'] = df['close_price'].pct_change() * df['position']

    # cum return
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1

    # net value
    df['daily_net_return'] = (1 + df['daily_return']).cumprod() - 1
    df['net_value'] = 1 * (1 + df['daily_net_return'])
    df = df.dropna()
    abs_return = df['net_value'].iloc[-1] - df['net_value'].iloc[0]
    annual_return = abs_return / time_delta_years
    net_value = df['net_value'].iloc[-1]

    # max draw down
    roll_max = df['net_value'].cummax()
    daily_draw_down = df['net_value'] / roll_max - 1.0
    max_daily_draw_down = abs((daily_draw_down.cummin().iloc[-1] * 100))

    # avg and std
    avg = df['daily_return'].mean()
    std = df['daily_return'].std()
    sharpe = np.sqrt(365) * (avg / std)
    calmar = (abs_return / time_delta_years) / max_daily_draw_down
    # print(df)
    print(sharpe)
    return df, transaction_time, transaction_time_average, max_daily_draw_down, sharpe, calmar, annual_return, net_value


def plot_figures(df, factor_name):
    # plot positions
    fig, ax = plt.subplots(dpi=1000, figsize=(14, 6))
    plt.plot(df['date'], df['close_price'])

    plt.xlabel("date")
    plt.ylabel("close price (continuous)")

    plt.xticks(rotation=45)
    ax.set_xticks([x for x in range(0, len(df['date']), 180)])

    plt.title('Long position (red) and short position (green)')

    y1 = np.arange(0, 1, 0.01)
    y2 = np.arange(-1, 0, 0.01)
    ax2 = ax.twinx()
    for i in range(len(df)):
        row = df.iloc[i]
        temp_date = row['date']
        if row['signal'] == 1:
            ax2.fill_betweenx(y1, temp_date, temp_date, color='red', alpha=.5)
        elif row['signal'] == 0:
            ax2.fill_betweenx(y2, temp_date, temp_date, color='green', alpha=.5)

    plt.savefig(f"result/position_plot_{factor_name}.png")
    plt.close()

    # plot net values
    fig, ax = plt.subplots(dpi=1000, figsize=(14, 6))
    plt.plot(df['date'], df['net_value'])

    plt.xlabel("date")
    plt.ylabel("net value (starting at 1)")

    plt.xticks(rotation=45)
    ax.set_xticks([x for x in range(0, len(df['date']), 180)])

    plt.title('net value (train set)')
    plt.savefig(f"result/net_value_plot_{factor_name}.png")
    plt.close()

    # plot PnL
    return None


if __name__ == "__main__":
    os.chdir("../../")
    factor_list = ['basis_spot', 'profit', 'consume_1', 'multi_factor']
    gap_day_list = [2, 7, 5, 5]
    factor_choice = 0  # 0: basis factor, 1: profit factor, 2: consumption factor, 3: multi_factor

    factor = factor_list[factor_choice]
    gap_day = gap_day_list[factor_choice]
    print("factor:", factor)
    print("gap_day:", gap_day)
    file_path = f"signal_data/signal_{factor}_{gap_day}.csv"

    data_signal, data_price = load_data(file_path)
    result_train, result_test = compute_position(data_signal, data_price, factor)

    result_train, transaction_time_train, transaction_time_average_train, max_daily_draw_down_train, \
        sharpe_train, calmar_train, annual_return_train, net_value_train = compute_return_statistics(result_train)

    result_test, transaction_time_test, transaction_time_average_test, max_daily_draw_down_test, \
        sharpe_test, calmar_test, annual_return_test, net_value_test = compute_return_statistics(result_test)

    df_result = pd.DataFrame({"total_transaction_times": [transaction_time_train, transaction_time_test],
                              "average_transaction_times": [transaction_time_average_train,
                                                            transaction_time_average_test],
                              "net_value": [net_value_train, net_value_test],
                              "annual_return": [annual_return_train, annual_return_test],
                              "max_draw": [max_daily_draw_down_train, max_daily_draw_down_test],
                              "sharpe": [sharpe_train, sharpe_test],
                              "calmar": [calmar_train, calmar_test]})
    print(df_result)
    df_result.to_csv(f"result/result_{factor}.csv")

    # plot positions and net value
    plot_figures(result_train, factor_name=factor)
    print("The result is saved at result/")
