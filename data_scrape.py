import pandas as pd
import os
import yfinance as yf

# Download elon tweets
os.system("snscrape --jsonl twitter-search 'from:elonmusk'> ./data/download/elon-tweets.json")

# Download stocks
tsla = yf.Ticker('TSLA')
twtr = yf.Ticker('TWTR')
btc = yf.Ticker('BTC-USD')
doge = yf.Ticker('DOGE-USD')
pd.DataFrame(tsla.history(period="max")).to_csv('./data/download/tesla.csv')
pd.DataFrame(twtr.history(period="max")).to_csv('./data/download/twitter.csv')
pd.DataFrame(btc.history(period="max")).to_csv('./data/download/bitcoin.csv')
pd.DataFrame(doge.history(period="max")).to_csv('./data/download/dogecoin.csv')