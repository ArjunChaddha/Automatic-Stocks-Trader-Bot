import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import pyEX as p
import pandas_datareader as web
from matplotlib import style
from nsetools import nse


# stocks = ['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BHARTIARTL.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GAIL.NS', 'GRASIM.NS', 'HDFC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'IOC.NS', 'IRCON.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHREECEM.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS' ]
stocksIown = ['WIPRO.NS', 'LUXIND.NS', 'DFMFOODS.NS', 'SUBROS.NS']
style.use('ggplot')

# start = dt.datetime(2019, 1, 1)
# end = dt.datetime(2020, 12, 30)
# df = web.DataReader('PSPPROJECT.NS', 'yahoo', start, end)
# df.to_csv('PSPPROJECT.NS')
#
for stock in stocksIown:
    start = dt.datetime(2019, 1, 1)
    end = dt.datetime(2021, 1, 6)
    df = web.DataReader(stock, 'yahoo', start, end)
    df.to_csv(stock)
    print(stock + " done")










