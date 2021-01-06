#OBV (On balance volume) code

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

#Calculate the On Balance Volume (OBV)
OBV = []
OBV.append(0)

#Loop through the data set (close price) from the second row (index 1) to the end of the data set
for i in range (1, len(df.Close)):
    if df.Close[i] > df.Close[i-1]:
        OBV.append(OBV[-1] + df.Volume[i])
    elif df.Close[i] < df.Close[i-1]:
        OBV.append(OBV[-1] - df.Volume[i])
    else:
        OBV.append(OBV[-1])

#Store the OBV and OBV Exponential moving average (EMA) into new columns
df['OBV'] = OBV
df['OBV_EMA'] = df['OBV'].ewm(span=20).mean()

#Create and plot the graph
plt.figure(figsize=(12.2, 4.5))
plt.plot(df['OBV'], label='OBV', color='orange')
plt.plot(df['OBV_EMA'], label='OBV_EMA', color='purple')
plt.xticks(rotation=45)
plt.title('OBV against OBV_EMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()

#Create a function to signal when to buy and sell the stock

#IF OBV > OBV_EMA, buy. IF OBV < OBV_EMA, then sell
def buy_sell(signal): #col1 = OBV, col2 = OBV_EMA
    sigPriceBuy=[]
    sigPriceSell=[]
    flag = -1

    for i in range(0, len(signal)):
        if signal['OBV'][i] > signal['OBV_EMA'][i] and flag != 1:
            sigPriceBuy.append(signal['Close'][i])
            sigPriceSell.append(np.nan)
            flag=1
        elif signal['OBV'][i] < signal['OBV_EMA'][i] and flag != 0:
            sigPriceSell.append(signal['Close'][i])
            sigPriceBuy.append(np.nan)
            flag = 0
        else:
            sigPriceBuy.append(np.nan)
            sigPriceSell.append(np.nan)

    return (sigPriceBuy, sigPriceSell)

#Call the function
x = buy_sell(df)
df['Buy_Signal_Price'] = x[0]
df['Sell_Signal_Price'] = x[1]

#Plot
plt.figure(figsize=(12.2, 4.5))
plt.plot(df['Close'], label='Close', alpha=0.35)
plt.scatter(df.index, df['Buy_Signal_Price'], label='Buy Signal', marker='^', color='green', alpha=1)
plt.scatter(df.index, df['Sell_Signal_Price'], label='Sell Signal', marker='v', color='red', alpha=1)
plt.xticks(rotation=45)
plt.title('OBV against OBV_EMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()





