import yfinance as yf
import pandas as pd

value = 0
data = yf.download("DFMFOODS.NS", start="2019-01-01")
print (data['Close'][-2])
print (data['Close'][-1])
value = value + (data['Close'][-1])*10

data = yf.download("LUXIND.NS", start="2019-01-01")
print (data['Close'][-2])
print (data['Close'][-1])
value = value + (data['Close'][-1])*2

data = yf.download("SUBROS.NS", start="2019-01-01")
print (data['Close'][-2])
print (data['Close'][-1])
value = value + (data['Close'][-1])*9

percentage_gain = ((value-10523.25)/10523.25)*100
money_gained = value-10523.25

print ("You had invested: 10523.25 on 30th December 2020")
print ("Now, the value of your stocks are: ",  value)
print ("Money gained: ", money_gained)
print ("Percentage gain: ",  percentage_gain)


data = yf.download("WIPRO.NS", start="2019-01-01")
print (data['Close'][-2])
print (data['Close'][-1])