# US-Stocks-Trading-Strategies
The goal of this repo is to design 3 simple trading strategies trading 29 US Tech Stocks (30m price data). There are 3 files:

- data.csv: Excel file containing 29 columns for the stocks and their 30m price.
- strategies_functions.py: File containing some functions called in the notebook - mainly for plotting
- main.ipynb: Notebook containing all the visualizations.

_______________________________________________________________________________________________________________________________
The notebook is composed of:

### **1. Data Analysis:**
    - Plots
    - Distributions
    - Q-Q/Box plots
    - Stationarity
    - Correlation
    - Autocorrelation
### **2. Trading Strategies:**
    - A. Technical Analysis based
    - B. Regression based
    - C. Dimensionality-reduction based
_______________________________________________________________________________________________________________________________

#### Expanding on the Trading Strategies:

The strategies are all benchmarked against Buy-and-Hold strategy. The total return as well as annualised Sharpe Ratio are calculated for both in order to have a better understanding of the performances.

_______________________________________________________________________________________________________________________________

**2.A Technical Analysis based**

The strategy implemented in the Dual Moving Average Crossover strategy. It consists in computing a long-term Moving Average (MA) and a short-term one. The two indicators have different rates of directions, and when they cross (crossover points), it indicates a buy/sell signal. The intuition behind this strategy can be explained in terms of momentum. Basically, the principle of momentum states that a price that is moving up (or down) during period t is likely to continue to move up (or down) in period t+1 unless evidence exists to the contrary. When the short-term MA moves above the long-term MA, this provides a lagged indicator that the price is moving upward relative to the historical price, which indicates a long position. In the contrary, it indicates a short position.

Below, a snapshot of a stock price, with the 2 MA (Green: Long-term, Orange: Short-term) with the crossover points indicating whether to Long or Short the stock.

![image](https://github.com/fernando2xV/US-Stocks-Trading-Strategies/assets/53314986/5019e429-f2ef-41fa-9e7e-0646a8e40405)

Finally, the plot of Market returns vs. Strategy returns (Technical Analysis):

![image](https://github.com/fernando2xV/US-Stocks-Trading-Strategies/assets/53314986/a4acba8b-c99a-46d6-833c-691f91bcd67a)

_______________________________________________________________________________________________________________________________

**2.B Regression based**

This strategy uses the Vector AutoRegressive model to forecast future returns. It works as follows:

1. Iterate through the data 1 timestep at a time

2. Fit the VAR model to the data available

3. At each timestep t, make predictions for the returns of the 29 stocks at timestep t+1

4. If predictions are bigger than a threshold -> BUY. If smaller than a threshold -> SELL. (threshold is optional, it's just to "only L/S if the prediction is large in magnitude"). Threshold can be set to 0.

Finally, the plot of Market returns vs. Strategy returns (Regression):

![image](https://github.com/fernando2xV/US-Stocks-Trading-Strategies/assets/53314986/93b706b1-db1b-47e3-b859-82b9a40a4f44)

_______________________________________________________________________________________________________________________________

**2.C Dimensionality-Reduction based**

This strategy uses Principal Component Analysis (PCA) to allocate weights of stocks in the portfolio. It analyses the relationship between multiple assets and uses the concept of eigenportfolio, which is a technique used to transform a set of correlated variables into a new set of uncorrelated variables called principal components.

It works as follows:

1. Data preprocessing
2. Covariance Matrix Calculation
3. Principal Components from PCA
4. Portfolio Allocation
5. [Not implemented] Portfolio Rebalancing

Finally, the plot of Market returns vs. Strategy returns (Different Eigenportfolios plotted):

![image](https://github.com/fernando2xV/US-Stocks-Trading-Strategies/assets/53314986/2c976a14-c9fa-4de4-bb7b-063ba47ce7e3)

The performance of the different Eigenportfolios + market:

![image](https://github.com/fernando2xV/US-Stocks-Trading-Strategies/assets/53314986/3bfde4f7-e0b2-4c27-b95c-7ef6b76e6623)

_______________________________________________________________________________________________________________________________

For further explanation, context and reasons of using these strategies, please refer to the main.ipynb.
