#Bollinger bands code

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Load the data
df = pd.read_csv('RELIANCE.csv')

#Make the date the index
df = df.set_index(pd.DatetimeIndex(df['Date']))

#visually show the price
plt.figure(figsize=(12.2, 4.5))
plt.plot(df['Adj Close'], label='Adj Close Price')
plt.xticks(rotation=45)
plt.title('Adj Close Price History')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#Calculate the simple moving average, standard deviation, upper band, and lower band
#Get 20 day time period
period = 20

#Calculate the simple moving average (SMA)
df['SMA'] = df['Close'].rolling(window=period).mean()

#Get the standard deviation
df['STD'] = df['Close'].rolling(window=period).std()

#Calculate the upper band
df['Upper'] = df['SMA'] + (df['STD']*2)

#Calculate the lower band
df['Lower'] = df['SMA'] - (df['STD']*2)

#Create a list of columns to keep
column_list = ['Close', 'SMA', 'Upper', 'Lower']

#Plot the data
df[column_list].plot(figsize=(12.2,4.5))
plt.title("Bollinger Band")
plt.ylabel("Price")
plt.show()

#Plot and shade the area between the two bollinger bands
#Get the figure and the figure size
fig = plt.figure(figsize=(12.2, 4.5))

#Add the subplot
ax = fig.add_subplot(1,1,1)

#Get the index values of the data frame
x_axis = df.index

#Plot and shade the area between the upper band and the lower band gray
ax.fill_between(x_axis, df['Upper'], df['Lower'], color='grey')

#Plot the closing price and moving average
ax.plot(x_axis, df['Close'], color='gold', lw=3, label = 'Close Price')
ax.plot(x_axis, df['SMA'], color='blue', lw=3, label = 'Simple Moving Average')

#Set the title and show the image
ax.set_title("Bollinger bands for stock")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
plt.xticks(rotation=45)
ax.legend()
plt.show()


#Create a new dataframe
new_df = df[period-1:]

#show new data
print(new_df)

#Create a function to get the buy and sell signals

def get_signal(data):
    buy_signal = []
    sell_signal = []

    for index in range(len(data['Close'])):
        if data['Close'][index] > data['Upper'][index]: #Then you should sell
            buy_signal.append(np.nan)
            sell_signal.append(data['Close'][index])

        elif data['Close'][index] < data['Lower'][index]: #Then you should buy
            sell_signal.append(np.nan)
            buy_signal.append(data['Close'][index])

        else:
            buy_signal.append(np.nan)
            sell_signal.append(np.nan)

    print (buy_signal)
    return buy_signal, sell_signal


#Create two new columns
new_df['Buy'] = get_signal(new_df)[0]
new_df['Sell'] = get_signal(new_df)[1]

#Plot all of the data

#Plot and shade the area between the two bollinger bands
#Get the figure and the figure size
fig = plt.figure(figsize=(12.2, 4.5))

#Add the subplot
ax = fig.add_subplot(1,1,1)

#Get the index values of the data frame
x_axis = new_df.index

#Plot and shade the area between the upper band and the lower band gray
ax.fill_between(x_axis, new_df['Upper'], new_df['Lower'], color='grey')

#Plot the closing price and moving average
ax.plot(x_axis, new_df['Close'], color='gold', lw=3, label = 'Close Price', alpha=0.5)
ax.plot(x_axis, new_df['SMA'], color='blue', lw=3, label = 'Simple Moving Average', alpha=0.5)
ax.scatter(x_axis, new_df['Buy'], color='green', lw=3, label='Buy', marker='^', alpha=1)
ax.scatter(x_axis, new_df['Sell'], color='red', lw=3, label='Sell', marker='v', alpha=1)
#Set the title and show the image
ax.set_title("Bollinger bands for stock")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
plt.xticks(rotation=45)
ax.legend()
plt.show()

