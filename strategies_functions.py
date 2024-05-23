import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.preprocessing import minmax_scale


# ####################                           ####################
# ####################                           ####################
# ####################                           ####################
# #################### [Signal 1 - MA Crossover] #################### 
# ####################                           ####################
# ####################                           ####################
# ####################                           ####################

def calculate_moving_averages(prices, short_window, long_window):
    # returns short and long moving averages
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()
    return short_ma, long_ma


# signal generator
def generate_signal_moving_averages_crossover(prices, short_ma, long_ma):
    signals = pd.DataFrame(index=prices.index, columns=prices.columns)
    previous_short_ma = short_ma.shift()
    # when short crosses above long, we long, and keep "longing" till it shorts. And viceversa.
    signals[(short_ma > long_ma)] = 1
    signals[(short_ma < long_ma)] = -1
    return signals


# visualize strategy
def visualize_moving_averages_crossover(prices, short_ma, long_ma, signals, column_name):
    plt.figure(figsize=(20, 10))
    plt.plot(prices.index, prices[column_name], label='Price')
    plt.plot(prices.index, short_ma[column_name], label='Short-term Moving Average', alpha=0.5)
    plt.plot(prices.index, long_ma[column_name], label='Long-term Moving Average', alpha=0.5)

    # we want to detect the point where it goes from -1-->1 and viceversa
    buy_signals = np.where((signals[column_name].shift(1) == -1) & (signals[column_name] == 1))[0]
    sell_signals = np.where((signals[column_name].shift(1) == 1) & (signals[column_name] == -1))[0]

    if len(buy_signals) > 0:
        plt.scatter(prices.index[buy_signals], prices[column_name][buy_signals], marker='^', color='g', label='Buy')
    if len(sell_signals) > 0:
        plt.scatter(prices.index[sell_signals], prices[column_name][sell_signals], marker='v', color='r', label='Sell')

    plt.xlabel('Date')

    # Customize x-axis tick placement and rotation
    plt.xticks(rotation=90)

    # Increase spacing between x-axis labels and plot
    plt.subplots_adjust(bottom=0.2)

    plt.ylabel('Price')
    plt.title(f'Stock Price and Moving Averages for {column_name}')
    plt.legend()
    plt.show()


def calculate_returns(returns, signals):
    """
    Calculate the log-returns of the trading strategy.

    Input:
    returns -- DataFrame containing stock log-returns
    signals -- DataFrame containing trading signals - 1's or -1's

    Output:
    investment_returns -- DataFrame containing log-returns for the strategy
    """
    returns = returns.iloc[1:, :]
    signals = signals.iloc[1:, :]
    dummy_idx = returns.index
    returns.index = signals.index

    investment_returns = returns.mul(signals.values)
    return investment_returns


# ###################                             ####################
# ###################                             ####################
# ###################                             ####################
# ################### [Signal 2 - VAR prediction] ################### 
# ###################                             ####################
# ###################                             ####################
# ###################                             ####################

# signal generator
def generate_signal_var(returns, lags, n_lags):
    """
    Iterates through the data and fits a VAR model using the last 'lags' datapoints to predict t+1 timestep ahead.
    """

    # initiate lists to append data to
    signals = []
    all_preds = []
    all_true = []
    signs_correctly_predicted = []

    # Iterate through the remaining rows of the data
    for i in tqdm(range(n_lags * lags, len(returns) - 1)):

        # data up until 't'
        # train_data = np.array(returns.iloc[:i, :])

        # data from 't-lags' to 't'
        train_data = np.array(returns.iloc[i - (n_lags * lags):i, :])

        # Get today's returns and date ('t')
        current_data = np.array(returns.iloc[i, :])

        # Model 1 - VAR
        model = sm.tsa.VAR(train_data)  # fit var process and do lag order selection
        model_fit = model.fit(lags, verbose=True)  # fit var model
        # print(model_fit.summary()) # only works if lags >= number of columns (29)

        # Make predictions for the next timestamp return
        predictions = model_fit.forecast(y=train_data, steps=1)

        # append predictions
        all_preds.append(predictions[0])  # store predictions
        all_true.append(np.array(returns.iloc[i + 1, :]))  # store true data

        # check how many of the 29 stocks is the sign correct
        signs_match = np.sign(predictions[0]) == np.sign(np.array(returns.iloc[i + 1, :]))
        signs_match = signs_match.astype(int)
        print(f"Correctly predicted the sign: {sum(signs_match)} / {len(returns.columns)}")

        # threshold to only make predictions if we have a 'strong' prediction
        threshold = 0.0
        if threshold != 0:
            buy_sell_signals = np.empty(len(predictions[0]))
            buy_sell_signals[:] = np.nan

        else:
            buy_sell_signals = np.ones(len(predictions[0]))

        # Compare predictions with current values to determine buy/sell signals
        buy_sell_signals[predictions[0] < -threshold] = -1  # all predictions that are positive, we "buy". Else, we sell
        buy_sell_signals[predictions[0] > threshold] = 1

        # Store the buy/sell signals for the current timestamp
        signals.append(buy_sell_signals)
        signs_correctly_predicted.append(sum(signs_match))

    signals = pd.DataFrame(data=signals,
                           index=returns.iloc[n_lags * lags:len(returns) - 1, :].index,
                           columns=returns.columns)
    signals = signals.ffill()  # needed because signals will be multiplied by each return

    all_preds = pd.DataFrame(data=all_preds,
                             index=returns.iloc[n_lags * lags:len(returns) - 1, :].index,
                             columns=returns.columns)
    all_true = pd.DataFrame(data=all_true,
                            index=returns.iloc[n_lags * lags:len(returns) - 1, :].index,
                            columns=returns.columns)
    print("")
    print("Buy Sell Signals: ")
    print(signals)
    return signals, all_preds, all_true, signs_correctly_predicted


# visualize strategy
def visualize_strategy_var(prices, signals, stock):
    """
    Visualize the trading strategy.

    Input:
    prices -- DataFrame containing stock prices
    signals -- DataFrame containing trading signals

    """
    plt.figure(figsize=(10, 6))

    # this is to plot all time series
    for col in [stock]:
        plt.plot(prices.index, prices[col], label=col)
        signals_reset = signals.reset_index(drop=True)

        plt.scatter(prices.index[signals[col].values == 1.0], prices[col][signals[col].values == 1.0], marker='^', color='g', label='Buy')
        plt.scatter(prices.index[signals[col].values == -1.0], prices[col][signals[col].values == -1.0], marker='v', color='r', label='Sell')
    plt.title('Stock Prices with Trading Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    # plt.xlim(prices.index[0], prices.index[99])
    plt.legend()
    plt.show()


def visualize_accuracy_predictions(signs_correctly_predicted):
    """
    helper function to visualize the accuracy of predictions
    """
    # Create a figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Plot the time series
    axs[0].plot(signs_correctly_predicted)
    axs[0].set_title('Number of correctly predicted stocks vs Timestamp')
    axs[0].set_xlabel('Index of prediction')
    axs[0].set_ylabel('Number of correctly predicted stocks')

    # Plot the histogram
    axs[1].hist(signs_correctly_predicted, bins=25)
    axs[1].set_title('Distribution of Number of correctly predicted stocks')
    axs[1].set_xlabel('Number of correctly predicted stocks')
    axs[1].set_ylabel('Frequency')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
