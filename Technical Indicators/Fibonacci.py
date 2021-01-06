import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib

df = pd.read_csv('RELIANCE.NS')
df = df.set_index(pd.DatetimeIndex(df['Date']))

# plt.figure(figsize=(12.2, 4.5))
# plt.plot(df.index, df['Close'], label='Closing Price', color='blue')
# plt.legend(loc='upper left')
# plt.show()

price_min = (df['Close'].min())
price_max = (df['Close'].max())

diff = price_max - price_min
level1 = price_max - 0.236 * diff
level2 = price_max - 0.382 * diff
half = price_max - 0.5 * diff
level3 = price_max - 0.618 * diff

# print("Level", "Price")
# print("0 ", price_max)
# print("0.236", level1)
# print("0.382", level2)
# print("0.618", level3)
# print("1 ", price_min)

fig = plt.figure(figsize=(12.2, 4.5))
x_axis = df.index
ax = fig.add_subplot(1,1,1)
ax.fill_between(x_axis, level1, price_min, color='grey', alpha=0.2)
ax.fill_between(x_axis, level2, level1, color='red', alpha=0.2)
plt.plot(df.index, df['Close'], label='Closing Price', color='blue')
ax.fill_between(x_axis, half, level2, color='orange', alpha=0.2)
ax.fill_between(x_axis, level3, half, color='blue', alpha=0.2)
ax.fill_between(x_axis, price_max, level3, color='green', alpha=0.2)
plt.ylabel("Price")
plt.xlabel("Date")
plt.legend(loc=2)
plt.show()