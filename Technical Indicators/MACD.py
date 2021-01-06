# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Loading data from csv file
df = pd.read_csv('RELIANCE.csv')

# Setting date as the index
df = df.set_index(pd.DatetimeIndex(df['Date']))

# Showing data from CSV file
#print(df)

# Visually showing the closing stock price
#plt.figure(figsize=(12.2, 4.5))
#plt.plot(df['Close'], label='Close')
#plt.xticks(rotation=45)
#plt.title('Close Price History')
#plt.xlabel('Date')
#plt.ylabel('Price')
#plt.show()

# Calculate the MACD and Signal Line Indicators
# Calculate the short term exponential moving average (EMA)
ShortEMA = df.Close.ewm(span=12, adjust=False).mean()

# Calculate the long term exponential moving average (EMA)
LongEMA = df.Close.ewm(span=26, adjust=False).mean()

# Calculate the MACD line
MACD = ShortEMA - LongEMA

# Calculate the signal line
Signal = MACD.ewm(span=9, adjust=False).mean()

Histogram = MACD - Signal

# Plot the chart
#plt.figure(figsize=(12.2, 4.5))
#plt.plot(df.index, MACD, label='TSLA MACD', color='red')
#plt.plot(df.index, Signal, label='Signal Line', color='blue')
#plt.xticks(rotation=45)
#plt.legend(loc='upper left')
#plt.show()

# Create new columns for the data
df['MACD'] = MACD
df['Signal Line'] = Signal
df['Histogram'] = Histogram

#Show the data
#print(df.head(5))


#Create a function to signal when to buy and sell a stock
def buy_sell(signal):
    Buy = []
    Sell = []
    Buy_Histogram = []
    Sell_Histogram = []

    # When MACD is greater than Signal line, and flag value is negative, it means a
    # crossover happened. Hence, append the closing price to the Buy list, and append
    # not a number to the sell list. Same thing happens when Signal > MACD
    for index in range(0, len(signal)):
        if signal['MACD'][index] > signal['Signal Line'][index] and signal['MACD'][index] > 0:
            if signal['MACD'][index-1] < signal['Signal Line'][index-1]:
                Buy.append(signal['Close'][index])
            else:
                Buy.append(np.nan)
        else:
            Buy.append(np.nan)

        if signal['Histogram'][index] > 0 and signal['Histogram'][index] < signal['Histogram'][index-1] < signal['Histogram'][index-2] < signal['Histogram'][index-3]  < signal['Histogram'][index-4]  < signal['Histogram'][index-5]:
            Sell_Histogram.append(signal['Close'][index])
        else:
            Sell_Histogram.append(np.nan)

        if signal['Histogram'][index] < 0 and signal['Histogram'][index] > signal['Histogram'][index - 1] > signal['Histogram'][index - 2] > signal['Histogram'][index-3] > signal['Histogram'][index-4] > signal['Histogram'][index-5]:
            Buy_Histogram.append(signal['Close'][index])
        else:
            Buy_Histogram.append(np.nan)

        if signal['MACD'][index] < signal['Signal Line'][index] and signal['MACD'][index] < 0:
            if signal['MACD'][index-1] > signal['Signal Line'][index-1]:
                Sell.append(signal['Close'][index])
            else:
                Sell.append(np.nan)
        else:
            Sell.append(np.nan)


    return (Buy, Sell, Buy_Histogram, Sell_Histogram)


# Create a buy and sell column to test this
a = buy_sell(df)
df['Buy_Histogram'] = a[2]
df['Sell_Histogram'] = a[3]
df['Buy_Signal_price'] = a[0]
df['Sell_Signal_price'] = a[1]


#Show the data
#pd.set_option("display.max_rows", None, "display.max_columns", None)
print(df)

#Visually show the stock buy and sell signals
plt.figure(figsize=(12.2,4.5))
plt.scatter(df.index, df['Buy_Signal_price'], color='green', label='Buy', marker= '^', alpha=1)
plt.scatter(df.index, df['Sell_Signal_price'], color='red', label='Sell', marker= 'v', alpha=1)
plt.scatter(df.index, df['Buy_Histogram'], color='orange', label='Buy Histogram', marker= '^', alpha=1)
plt.scatter(df.index, df['Sell_Histogram'], color='purple', label='Sell Histogram', marker= 'v', alpha=1)
plt.plot(df['Close'], label='Close Price', alpha=0.35)
plt.title('Close Price Buy and Sell Signals')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend(loc='upper left')
plt.show()
