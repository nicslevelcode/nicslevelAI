import pandas as pd
import matplotlib.pyplot as plt
import talib as talib

tsla_df = pd.read_csv('C:/Users/nicho/Desktop/TSLA.csv')

tsla_df['Adj 5 days future']=tsla_df['Adj Close'].shift(-5)
tsla_df['Adj 5 days future pct']=tsla_df['Adj 5 days future'].pct_change(5)
tsla_df['Adj Close 5 days future pct']=tsla_df['Adj Close'].pct_change(5)

print(tsla_df.head(10))

plt.scatter(tsla_df['Adj 5 days future pct'], tsla_df['Adj Close 5 days future pct'])
plt.show()

corr=tsla_df[['Adj 5 days future pct','Adj Close 5 days future pct']].corr()
print(corr)

feature_names = ['Adj Close 5 days future pct']  # a list of the feature names for later
# Create moving averages and rsi for timeperiods of 14, 30, 50, and 200
for n in [14, 30, 50, 200]:
    # Create the moving average indicator and divide by Adj Close
    tsla_df['ma' + str(n)] = talib.SMA(tsla_df['Adj Close'].values,
                              timeperiod=n) / tsla_df['Adj Close']
    # Create the RSI indicator
    tsla_df['rsi' + str(n)] = talib.RSI(tsla_df['Adj Close'].values, timeperiod=n)
    
    # Add rsi and moving average to the feature name list
    feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]
   
print(feature_names)
