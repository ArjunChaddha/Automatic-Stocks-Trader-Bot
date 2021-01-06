#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#load the data
df = pd.read_csv('RELIANCE.csv')

#Show the data
#print(df)

#make the date the index
df = df.set_index(pd.DatetimeIndex(df['Date']))

#show the new data
print(df)

#visually show the price
#plt.figure(figsize=(12.2, 4.5))
#plt.plot(df['Adj Close'], label='Adj Close Price')
#plt.xticks(rotation=45)
#plt.title('Adj Close Price History')
#plt.xlabel('Date')
#plt.ylabel('Price')
#plt.show()

#Prepare the data to calculate the RSI
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

#Put it all together

#Create a new data frame
new_df = pd.DataFrame()
new_df['Adj Close Price'] = df['Adj Close']
new_df['RSI'] = RSI
print(new_df)

#Visually show the adjusted close price and RSI
#Plot the adjusted close price
plt.figure(figsize=(12.2, 4.5))
plt.plot(new_df.index, new_df['Adj Close Price'])
plt.title('Adj Close Price History')
plt.legend(new_df.columns.values, loc = 'upper left')
plt.show()

#Plot the corresponding RSI values and the significant levels
plt.figure(figsize=(12.2, 4.5))
plt.title('RSI Plot')
plt.plot(new_df.index, new_df['RSI'])
plt.axhline(0, linestyle='--', alpha = 0.5, color = 'gray')
plt.axhline(10, linestyle='--', alpha = 0.5, color = 'orange')
plt.axhline(20, linestyle='--', alpha = 0.5, color = 'green')
plt.axhline(30, linestyle='--', alpha = 0.5, color = 'red')
plt.axhline(70, linestyle='--', alpha = 0.5, color = 'red')
plt.axhline(80, linestyle='--', alpha = 0.5, color = 'green')
plt.axhline(90, linestyle='--', alpha = 0.5, color = 'orange')
plt.axhline(100, linestyle='--', alpha = 0.5, color = 'gray')
plt.show()

#Make the buy/sell thing on the graph based on RSI crossovers

def buy_sell(signal):
    buy = []
    sell = []
    flag = -1

    # When RSI is greater than 70, and before it, it was less than 70, it means a
    # crossover happened. Hence, append the closing price to the Buy list, and append
    # not a number to the sell list. Same thing happens when RSI < 30
    for index in range(0, len(signal)):
        if signal['RSI'][index] < 30:
            sell.append(np.nan)
            if signal['RSI'][index-1] > 30:
                buy.append(signal['Adj Close Price'][index])
            else:
                buy.append(np.nan)

        elif signal['RSI'][index] > 80:
            buy.append(np.nan)
            if signal['RSI'][index-1] < 80:
                sell.append(signal['Adj Close Price'][index])
            else:
                sell.append(np.nan)

        else:
            buy.append(np.nan)
            sell.append(np.nan)

    return buy, sell


# Create a buy and sell column to test this
a = buy_sell(new_df)
df['Buy_Signal_price'] = a[0]
df['Sell_Signal_price'] = a[1]


#Show the data
#pd.set_option("display.max_rows", None, "display.max_columns", None)
#print(df)

#Visually show the stock buy and sell signals
plt.figure(figsize=(12.2,4.5))
plt.scatter(df.index, df['Buy_Signal_price'], color='green', label='Buy', marker= '^', alpha=1)
plt.scatter(df.index, df['Sell_Signal_price'], color='red', label='Sell', marker= 'v', alpha=1)
plt.plot(df['Adj Close'], label='Close Price', alpha=0.35)
plt.title('Close Price Buy and Sell Signals')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend(loc='upper left')
plt.show()
