#Combination of 3 technical indicators: Bollinger bands, RSI, and OBV to determine stock prices

#TO-DO -
#When the close price is kinda riding the upper BB and still going up, avoid selling at that time.
#FIBONAACCICIII
#FIX BB SQUEEZE VALUE (Make it percentage)
#And remove any i+ stuff to remove errors
#Riding downtrend is problem - u can make it so if there r alr 2 buycalls then no more wil be given - INDUSLDBK STOCK
#If its a continous downtrend, then it just doesnt give any sell calls - ADD STOP LOSS KID - IOC STOCK
#Not giving a sell signal for the next 10 days after BB squeeze and upmove - has flaws - SBILIFE stock
#WHen there is a very steep downtrend, just instantly sell (find a way to do dis) - BIRLAPREC STOCK

#ITC
# INDUSNDBANK
# MARUTI
# ONGC
# HEROMOTOCO
# GAIL


#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
plt.style.use('fivethirtyeight')

#Load the data
df = pd.read_csv('RELIANCE1.csv')

band_width_value = 0.24

#Make the date the index
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

#OBV CALCULATIONS
OBV = []
OBV.append(0)
for i in range (1, len(df.Close)): #Loop through the data set (close price) from the second row (index 1) to the end of the data set
    if df.Close[i] > df.Close[i-1]:
        OBV.append(OBV[-1] + df.Volume[i])
    elif df.Close[i] < df.Close[i-1]:
        OBV.append(OBV[-1] - df.Volume[i])
    else:
        OBV.append(OBV[-1])

#MACD CALCULATIONS
ShortEMA = df.Close.ewm(span=12, adjust=False).mean()
LongEMA = df.Close.ewm(span=26, adjust=False).mean()
MACD = ShortEMA - LongEMA
Signal = MACD.ewm(span=9, adjust=False).mean()
Histogram = MACD - Signal

#DEMA CALCULATIONS
def DEMA(data, time_period, column):
    EMA = data[column].ewm(span=time_period, adjust=False).mean()
    DEMA = (2 * EMA) - (EMA.ewm(span=time_period, adjust=False).mean())
    return DEMA



#MAKE A NEW DATA FRAME TO STORE ALL THIS
new_df = pd.DataFrame()
new_df['Close'] = df['Close']
new_df['Middle_band'] = Middle_band
new_df['STD'] = STD
new_df['Upper_band'] = Upper_band
new_df['Lower_band'] = Lower_band
new_df['Band_width'] = Bollinger_Band_Width
new_df['RSI'] = RSI
new_df['OBV'] = OBV
new_df['MACD'] = MACD
new_df['Signal Line'] = Signal
new_df['Histogram'] = Histogram
new_df['DEMA_short'] = DEMA(new_df, 20, 'Close')
new_df['DEMA_long'] = DEMA(new_df, 50, 'Close')
new_df['Open'] = df['Open']

#Plot fht BB_Band_Width
#plt.figure(figsize=(12.2, 4.5))
#plt.title('BB Band Width Plot')
#plt.plot(new_df.index, new_df['Band_width'])
#plt.show()


#GENERATE BUY CALLS WHEN BB PRICE BREAKS AND CLOSES ABOVE MIDDLE BAND AND RSI > 50
#GENERATE SELL CALLS WHEN PRICE CLOSES ABOVE THE UPPER BB BAND
#GENERATE BUY CALL WHEN PRICE CLOSES BELOW THE LOWER BB BAND

def buy_call (dataframe):
    buy_signal = []
    sell_signal = []

    for index in range(len(dataframe['Close'])):
        if dataframe['Close'][index] > dataframe['Upper_band'][index] and dataframe['RSI'][index] > 70:
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



#PLOT THESE BUY CALLS ON THE GRAPH
new_df['Buy_Signal'] = buy_call(new_df)[0]
new_df['Sell_Signal'] = buy_call(new_df)[1]
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199
new_df['stop_margin'] = new_df['Buy_Signal']* 0.1 #10% RETURNS


#Function to get the index of any value you input
def getIndex(dfCol, val):
    i = 0
    while i < len(dfCol):
        if dfCol[i] == val:
            return i
        else:
            i += 1
    return None

new_df['Sell'] = np.nan

#To get the target price (sell)
for B in new_df['Buy_Signal']:
    flag = 1
    if not math.isnan(B):
        index = getIndex(new_df['Buy_Signal'], B)

        for j in range (index, len(new_df)):
            if new_df['Close'][j] > B + new_df['stop_margin'][index] and flag == 1:
                new_df['Sell'][j] = new_df['Close'][j]
                flag = 0



#To get the buy signal again if the price does not go below middle_BB 5 days after the target price
new_df ['Buy'] = np.nan
for h in new_df['Sell']:
    flag = 1
    if not math.isnan(h):
        index = getIndex(new_df['Sell'], h)
        numDaysAhead = 5
        condition = True
        if len(new_df['Close']) - index > 6:
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
                new_df['Buy'][index + (len(new_df['Close']) - i)] =  new_df['Close'][index + (len(new_df['Close']) - i)]






#THis is to combine the 2 buy and sell lists with each other to create 2 final lists: Final_Buy and Final_Sell
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
    # print("i is: ", i)
    i += 1



#THIS PART GETS THE VALUE OF BB SQUEEZE
new_df['BB_Squeeze'] = np.nan
threshold_days = 30
max_breaks = 0


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

#print (new_df)

#This part gets the index of the BB Squeeze
new_df['test'] = np.nan
f = 0
sequence = 0
index = 0
while f < len (new_df['BB_Squeeze']):
    if not math.isnan(new_df['BB_Squeeze'][f]):
        flag = 1
        sequence += 1
    if math.isnan(new_df['BB_Squeeze'][f]):
        flag = 0
        if sequence > 0:
            index += getIndex(new_df['BB_Squeeze'][index:], (29+sequence))
            new_df['test'][index] = sequence
            flag = 1
        sequence = 0
    f += 1


#Check the closing price 5 days before that last BB squeeze value and if it goes above the top bracket, then buy
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


#CALCULATE THE RETURN ON INVESTMENT, ETC.
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
print (new_df)

print ('We gained:' , money)
print ("Money invested: ", investment)
print ("Now you have: ", final_money)
print ("Percentage return: ", percentage_return, '%')
print ("number of stocks in hand: ", number_of_stocks)


# After the sell call is generated (through TP or crossing upper BB), if the price doesnt go below middle_BB in the next 5 days, BUY AGAIN
#Take the vvalues of Sell which are not nan.

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

print (new_df)





