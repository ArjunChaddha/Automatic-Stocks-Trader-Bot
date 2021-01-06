# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Loading data from csv file
df = pd.read_csv('BRITANNIA1.csv')

# Setting date as the index
df = df.set_index(pd.DatetimeIndex(df['Date']))

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

# Plot the chart
#plt.figure(figsize=(12.2, 4.5))
#plt.plot(df.index, MACD, label='TSLA MACD', color='red')
#plt.plot(df.index, Signal, label='Signal Line', color='blue')
#plt.xticks(rotation=45)
#plt.legend(loc='upper left')
#plt.show()


#CALCULATE RSI
#Get the difference in price from the previous day
delta = df['Adj Close'].diff(1)
#print(delta)

#Get rid of NaN
delta = delta.dropna()
print(delta)

#Get the positive gains (up) and the negative gains (down)
up = delta.copy()
down = delta.copy()

up[up < 0] = 0
down[down > 0] = 0

#Get the time period
period = 14

#calculate the average gain and loss
AVG_Gain = up.rolling(window=period).mean()
AVG_Loss = abs(down.rolling(window=period).mean())

#Calculate the Relative Strength (RS)
RS = AVG_Gain / AVG_Loss

#Calculate the RSI
RSI = 100.0 - (100.0/(1.0 + RS))

#Show the RSI visually
#plt.figure(figsize=(12.2, 4.5))
#RSI.plot()
#plt.show()




# Create new columns for the data
new_df = pd.DataFrame()
new_df['MACD'] = MACD
new_df['Signal Line'] = Signal
new_df['Close'] = df['Close']
new_df['RSI'] = RSI


#Visually show the adjusted close price and RSI
#Plot the adjusted close price
#plt.figure(figsize=(12.2, 4.5))
#plt.plot(new_df.index, new_df['Close'])
#plt.title('Close Price History')
#plt.legend(new_df.columns.values, loc = 'upper left')
#plt.show()

#Plot the corresponding RSI values and the significant levels
#plt.figure(figsize=(12.2, 4.5))
#plt.title('RSI Plot')
#plt.plot(new_df.index, new_df['RSI'])
#plt.axhline(0, linestyle='--', alpha = 0.5, color = 'gray')
#plt.axhline(10, linestyle='--', alpha = 0.5, color = 'orange')
#plt.axhline(20, linestyle='--', alpha = 0.5, color = 'green')
#plt.axhline(30, linestyle='--', alpha = 0.5, color = 'red')
#plt.axhline(70, linestyle='--', alpha = 0.5, color = 'red')
#plt.axhline(80, linestyle='--', alpha = 0.5, color = 'green')
#plt.axhline(90, linestyle='--', alpha = 0.5, color = 'orange')
#plt.axhline(100, linestyle='--', alpha = 0.5, color = 'gray')
#plt.show()


#Create a function to signal when to buy and sell a stock
def buy_sell(signal):
    Buy = []
    Sell = []
    flag = -1

    # When MACD is greater than Signal line, and flag value is negative, it means a
    # crossover happened. Hence, append the closing price to the Buy list, and append
    # not a number to the sell list. Same thing happens when Signal > MACD
    for index in range(0, len(signal)):
        for i in range (0, 5):
            if signal['MACD'][index] > signal['Signal Line'][index] and signal['MACD'][index] > 0 and signal['RSI'][index-i] < 40:
                if signal['RSI'][index-i-1] > 40:
                    buy_signal = signal['Close'][index]
                else:
                    buy_signal = np.nan

            elif signal['MACD'][index] < signal['Signal Line'][index] and signal['MACD'][index] < 0 and (signal['RSI'][index-i] > 60):
                if signal['RSI'][index-i-1] < 60:
                    sell_signal = signal['Close'][index]
                else:
                    sell_signal = np.nan

            else:
                buy_signal = np.nan
                sell_signal = np.nan

        Buy.append(buy_signal)
        Sell.append(sell_signal)

    return (Buy, Sell)



# Create a buy and sell column to test this
a = buy_sell(new_df)
new_df['Buy_Signal_price'] = a[0]
new_df['Sell_Signal_price'] = a[1]

#Show the data
#pd.set_option("display.max_rows", None, "display.max_columns", None)
print(new_df)

#Visually show the stock buy and sell signals
plt.figure(figsize=(12.2,4.5))
plt.scatter(new_df.index, new_df['Buy_Signal_price'], color='green', label='Buy', marker= '^', alpha=1)
plt.scatter(new_df.index, new_df['Sell_Signal_price'], color='red', label='Sell', marker= 'v', alpha=1)
plt.plot(df['Close'], label='Close Price', alpha=0.35)
plt.title('Close Price Buy and Sell Signals')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend(loc='upper left')
plt.show()







