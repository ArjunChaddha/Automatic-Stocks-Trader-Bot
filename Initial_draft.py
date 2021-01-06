#threshold_days for the BB squeeze



#This bot uses a variety of technical indicators to generate buy and sell calls of any stock.

#Importing the needed libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
plt.style.use('fivethirtyeight')


#Loading the data and making the date the index
df = pd.read_csv('BAJFINANCE.csv')
df = df.set_index(pd.DatetimeIndex(df['Date']))


#BOLLINGER BANDS CALCULATIONS
Middle_band = df['Close'].rolling(window=20).mean() #MIDDLE BAND
STD = df['Close'].rolling(window=20).std() #STD DEVIATION
Upper_band = Middle_band + (STD*2) #UPPER BAND
Lower_band = Middle_band - (STD*2) #UPPER BAND
Bollinger_Band_Width = (Upper_band - Lower_band)/Middle_band #BOLLINGER BAND WIDTH


#RSI CALCULATIONS
delta = df['Close'].diff(1) #get the difference between closing prices
delta = delta.dropna()
up = delta.copy() #Get positive and negative gains/losses between 2 closing prices
down = delta.copy()
up[up < 0] = 0
down[down > 0] = 0
AVG_Gain = up.rolling(window=14).mean() #Calculate average gain and loss
AVG_Loss = abs(down.rolling(window=14).mean())
RS = AVG_Gain / AVG_Loss #Calculate relative strength
RSI = 100.0 - (100.0/(1.0 + RS)) #Calculate RSI


#DEMA CALCULATIONS
def DEMA(data, time_period, column):
    EMA = data[column].ewm(span=time_period, adjust=False).mean()
    DEMA = (2 * EMA) - (EMA.ewm(span=time_period, adjust=False).mean())
    return DEMA


#MAKE A NEW DATA FRAME TO STORE ALL THIS
new_df = pd.DataFrame()
new_df['Close'] = df['Close']
new_df['Open'] = df['Open']
new_df['Middle_band'] = Middle_band
new_df['STD'] = STD
new_df['Upper_band'] = Upper_band
new_df['Lower_band'] = Lower_band
new_df['Band_width'] = Bollinger_Band_Width
new_df['RSI'] = RSI
new_df['DEMA_short'] = DEMA(new_df, 20, 'Close')
new_df['DEMA_long'] = DEMA(new_df, 50, 'Close')


#Function to get the index of any value you input
def getIndex(dfCol, val):
    i = 0
    while i < len(dfCol):
        if dfCol[i] == val:
            return i
        else:
            i += 1
    return None

#THIS FUNCTION DOES THE FOLLOWING:
#Generate buy calls when BB price breaks and closes above middle band and RSI > 50
#Generate sell calls when price closes above the upper BB band and RSI > 70
#Generate buy calls when price closes below the lower BB band and RSI < 30
def buy_or_sell_call (dataframe):
    buy_signal = []
    sell_signal = []

    for index in range(len(dataframe['Close'])):
        if dataframe['Close'][index] > dataframe['Upper_band'][index] and dataframe['RSI'][index] > 72:
            sell_signal.append(dataframe['Close'][index])
            buy_signal.append(np.nan)

        elif dataframe['Close'][index] > dataframe['Middle_band'][index] and dataframe['RSI'][index] > 50:
            sell_signal.append(np.nan)
            if dataframe['Close'][index-1] < dataframe['Middle_band'][index-1] and dataframe['RSI'][index-1] < 50:# Then you should buy
                buy_signal.append(dataframe['Close'][index])
            else:
                buy_signal.append(np.nan)

        elif dataframe['Close'][index] < dataframe['Lower_band'][index] and dataframe['RSI'][index] < 30:
            sell_signal.append(np.nan)
            buy_signal.append(dataframe['Close'][index])

        else:
            buy_signal.append(np.nan)
            sell_signal.append(np.nan)

    return buy_signal, sell_signal



#Add new columns to our dataframe indicating the buy and sell signals we just got:
new_df['Buy_Signal'] = buy_or_sell_call(new_df)[0]
new_df['Sell_Signal'] = buy_or_sell_call(new_df)[1]
new_df['stop_margin'] = new_df['Buy_Signal'] * 0.25 #10% RETURNS


#The following block of code gives a sell signal every time you meet your target price (10% of the price you bought at)
new_df['Sell'] = np.nan
for Buy_Price in new_df['Buy_Signal']:
    flag = 1
    if not math.isnan(Buy_Price):
        index = getIndex(new_df['Buy_Signal'], Buy_Price)
        for j in range (index, len(new_df)):
            if new_df['Close'][j] > Buy_Price + new_df['stop_margin'][index] and flag == 1:
                new_df['Sell'][j] = new_df['Close'][j]
                flag = 0


#This block of code gives another buy signal if the closing price doesnt go below  middle BB 2 days after selling through target price
new_df['Buy'] = np.nan
for Sell_Price in new_df['Sell']:
    flag = 1
    if not math.isnan(Sell_Price):
        index = getIndex(new_df['Sell'], Sell_Price)
        numDaysAhead = 2
        condition = True
        if len(new_df['Close']) - index > (numDaysAhead+1):
            for i in range (index, index+numDaysAhead+1):
                if new_df['Close'][i] > new_df['Middle_band'][i]:
                    condition = condition and True
                else:
                    condition = condition and False
            if condition == True:
                new_df['Buy'][index + numDaysAhead] = new_df['Close'][index + numDaysAhead]
        else:
            for i in range (index, len(new_df['Close'])):
                if new_df['Close'][i] > new_df['Middle_band'][i]:
                    condition = condition and True
                else:
                    condition = condition and False
            if condition == True:
                new_df['Buy'][index + (len(new_df['Close']) - i-1)] =  new_df['Close'][index + (len(new_df['Close']) - i-1)]




#This is to combine the 2 buy and sell lists with each other to create 2 final lists: Final_Buy and Final_Sell
new_df['Final_Buy'] = np.nan
new_df['Final_Sell'] = np.nan
i = 0
while i < len(new_df['Buy_Signal']):
    if math.isnan(new_df['Buy_Signal'][i]) and math.isnan(new_df['Buy'][i]):
        new_df['Final_Buy'][i] = np.nan
    elif not math.isnan(new_df['Buy_Signal'][i]):
        new_df['Final_Buy'][i] = new_df['Buy_Signal'][i]
    else:
        new_df['Final_Buy'][i] = new_df ['Buy'][i]

    if math.isnan(new_df['Sell_Signal'][i]) and math.isnan(new_df['Sell'][i]):
        new_df['Final_Sell'][i] = np.nan
    elif not math.isnan(new_df['Sell_Signal'][i]):
        new_df['Final_Sell'][i] = new_df['Sell_Signal'][i]
    else:
        new_df['Final_Sell'][i] = new_df ['Sell'][i]
    i += 1



#The following block of code gets the value of BB squeeze
new_df['BB_Squeeze'] = np.nan

threshold_days = 30
max_breaks = 0
band_width_value = 0.24
i = 0
consecutive = 0
breaks = 0

while i < len(new_df):
    if new_df['Band_width'][i] < band_width_value:
        consecutive += 1
    else:
        breaks += 1
        if breaks > max_breaks:
            consecutive=0
            breaks=0
    if consecutive >= threshold_days:
        new_df['BB_Squeeze'][i] = consecutive
    i += 1


#This part gets the index of the BB Squeeze
new_df['test'] = np.nan
f = 0
sequence = 0
index = 0
while f < len(new_df['BB_Squeeze']):
    if not math.isnan(new_df['BB_Squeeze'][f]):
        flag = 1
        sequence += 1
    if math.isnan(new_df['BB_Squeeze'][f]):
        flag = 0
        if sequence > 0:
            index += getIndex(new_df['BB_Squeeze'][index:], ((threshold_days-1)+sequence))
            new_df['test'][index] = sequence
            flag = 1
        sequence = 0
    f += 1



#This code checks the closing price before the last BB squeeze value. If it goes above the top bracket, then buy
#If it goes below the bottom bracket, then sell
days_to_check_closing_price = 9
days_not_to_generate_buy_or_sell_call = 16

for value in new_df['test']:
    if not math.isnan(value):
        index = getIndex(new_df['test'], value)

        for i in range(index - days_to_check_closing_price, index):
            if new_df['Close'][i] > new_df['Upper_band'][i]:
                new_df['Final_Buy'][i] = new_df['Close'][i]
                if (len(new_df['Close']) - i) > days_not_to_generate_buy_or_sell_call:
                    for days in range(0, days_not_to_generate_buy_or_sell_call):
                        new_df['Final_Sell'][i + days] = np.nan
                else:
                    for days in range(0, (len(new_df['Close'])-i)):
                        new_df['Final_Sell'][i + days] = np.nan

            if new_df['Close'][i] < new_df['Lower_band'][i]:
                new_df['Final_Sell'][i] = new_df['Close'][i]
                if (len(new_df['Close']) - i) > days_not_to_generate_buy_or_sell_call:
                    for days in range(0, days_not_to_generate_buy_or_sell_call):
                        new_df['Final_Buy'][i + days] = np.nan
                else:
                    for days in range(0, (len(new_df['Close'])-i)):
                        new_df['Final_Buy'][i + days] = np.nan


#This part calculates the return on investment, percentage gain, etc.
money = 0
number_of_stocks = 0
investment = 0
a = 0
invests = []
while a < (len(new_df['Buy'])):
    if not math.isnan(new_df['Final_Buy'][a]):
        money = money - new_df['Final_Buy'][a]
        investment = investment + new_df['Final_Buy'][a]
        number_of_stocks = number_of_stocks + 1
        invests.append(money)

    elif not math.isnan(new_df['Final_Sell'][a]):
        money = money + (new_df['Final_Sell'][a] * number_of_stocks)
        number_of_stocks = 0
        invests.append(money)

    a += 1

if number_of_stocks > 0:
    money = money + number_of_stocks * new_df['Close'][-1]
    number_of_stocks = 0

investment = abs(min(invests))
percentage_return = (money/investment)*100
final_money = investment + money

print ('We gained:' , money)
print ("Money invested: ", investment)
print ("Now you have: ", final_money)
print ("Percentage return: ", percentage_return, '%')
print ("number of stocks in hand: ", number_of_stocks)


#This part simply generates a plot of the closing price, with all the buy and sell signals.

fig = plt.figure(figsize=(12.2, 4.5))
ax = fig.add_subplot(1,1,1)
x_axis = new_df.index
ax.fill_between(x_axis, new_df['Upper_band'], new_df['Lower_band'], color='grey')
ax.plot(x_axis, new_df['Close'], color='gold', lw=3, label = 'Close Price', alpha=0.5) #Plot closing price and moving average
ax.plot(x_axis, new_df['Middle_band'], color='blue', lw=3, label = 'Simple Moving Average', alpha=0.5)
ax.scatter(x_axis, new_df['Final_Buy'], color='green', lw=3, label='Buy', marker='^', alpha=1)
ax.scatter(x_axis, new_df['Final_Sell'], color='red', lw=3, label='Sell', marker='v', alpha=1)
ax.set_title("Bollinger bands for stock test")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
plt.xticks(rotation=45)
ax.legend()
plt.show()

