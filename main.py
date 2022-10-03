import pandas as pd
import yfinance as yf
from datetime import timedelta
'''
Action that being used:
MGLU3.SA
BBAS3.SA
WEGE3.SA
PETR3.SA
'''


def data(name):
    ticker = yf.Ticker(name)
    df = pd.DataFrame(ticker.history(period='5y'))
    df.reset_index(inplace=True)
    df_data = pd.DataFrame({'Date': pd.date_range(df['Date'].iloc[-1] + timedelta(days=1), periods=30)})
    df = df.append(df_data)
    df.fillna(0, inplace=True)
    return df


