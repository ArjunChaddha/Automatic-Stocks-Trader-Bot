import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
plt.style.use('fivethirtyeight')
stocks = ['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BHARTIARTL.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GAIL.NS', 'GRASIM.NS', 'HDFC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'IOC.NS', 'IRCON.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHREECEM.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS' ]

#Loading the data and making the date the index
def percentage_return (stock):
    df = pd.read_csv(stock)
    df = df.set_index(pd.DatetimeIndex(df['Date']))

    # print(df['Close'][-1])
    # print(df['Close'][0])

    return_of_stock = (df['Close'][-1]) - (df['Close'][0])
    percentage_return = (return_of_stock / (df['Close'][0])) * 100

    print(percentage_return)

    return percentage_return, return_of_stock

stock_list = []
percentage_returns = []

for stock in stocks:
    percentage_return(stock)
    stock_list.append(stock)
    excel = pd.DataFrame()
    excel['Stocks'] = stock_list
    percentage_returns.append(percentage_return(stock)[0])
    excel['Percentage Returns'] = percentage_returns
    excel.to_excel(r'/Users/arjun/Desktop/testing without bot.xlsx', index=False, header=True)



