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
#print(df)

#visually show the price
#plt.figure(figsize=(12.2, 4.5))
#plt.plot(df['Adj Close'], label='Adj Close Price')
#plt.xticks(rotation=45)
#plt.title('Adj Close Price History')
#plt.xlabel('Date')
#plt.ylabel('Price')
#plt.show()

#Create a function to calculate the Double exponential Moving Average
def DEMA(data, time_period, column):

    #Calculate the exponential moving average for some time period
    EMA = data[column].ewm(span=time_period, adjust=False).mean()

    #Calculate DEMA
    DEMA = (2 * EMA) - (EMA.ewm(span=time_period, adjust=False).mean())

    return DEMA

#Store the short term DEMA (20 day period) and the long term DEMA (50 day period)
df['DEMA_short'] = DEMA(df, 20, 'Close')
df['DEMA_long'] = DEMA(df, 50, 'Close')

#PLot the chart
#Create a list of columns to keep
#column_list = ['DEMA_short', 'DEMA_long', 'Close']
#df[column_list].plot(figsize=(12.2,6.4))
#plt.title("Moving average")
#plt.ylabel("Price")
#plt.xlabel("Date")
#plt.show()

#Create a function to buy and sell the stock
def DEMA_strategy(data):
    buy_list = []
    sell_list = []
    flag = False

    #Loop through the data
    for i in range (0, len(data)):
        if data['DEMA_short'][i] > data['DEMA_long'][i] and flag == False:
            buy_list.append(data['Close'][i])
            sell_list.append(np.nan)
            flag = True
        elif data['DEMA_short'][i] < data['DEMA_long'][i] and flag == True:
            sell_list.append(data['Close'][i])
            buy_list.append(np.nan)
            flag = False
        else:
            buy_list.append(np.nan)
            sell_list.append(np.nan)

    #Store the buy and sell signals/lists into the data set
    data['Buy'] = buy_list
    data['Sell'] = sell_list

#Run the strategy to get the buy and sell signals
DEMA_strategy(df)

#Visually show the stock buy and sell signals
plt.figure(figsize=(12.2, 4.5))
plt.scatter(df.index, df['Buy'], color='green', marker='^', label='Buy signal', alpha=1)
plt.scatter(df.index, df['Sell'], color='red', marker = 'v', label='Sell Signal', alpha=1)
plt.plot(df['Close'], label='Close Price', alpha=0.35)
plt.plot(df['DEMA_short'], label='Short term DEMA', alpha=0.35)
plt.plot(df['DEMA_long'], label='Long term DEMA', alpha=0.35)
plt.xticks(rotation=45)
plt.title("Close Price Buy and Sell Signals")
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Price', fontsize = 18)
plt.show()


