import pandas as pd
from collections import Counter
import re
import nltk
import itertools

elon_tweets = pd.read_json('./data/download/elon-tweets.json', lines=True) # need lines = True else trailing data error

tesla_stock = pd.read_csv('./data/download/tesla.csv')
twitter_stock = pd.read_csv('./data/download/twitter.csv')
bitcoin = pd.read_csv('./data/download/bitcoin.csv')
dogecoin = pd.read_csv('./data/download/dogecoin.csv')
stock_dfs = [tesla_stock,twitter_stock,bitcoin,dogecoin]

## Elon tweets dataframe
# Save all columns from original dataframe
all_elon_tweets_columns = elon_tweets.columns
# Define wanted columns
# For expansion: we could look at the mentioned/ replied to users
column_list = ['date','content','replyCount','retweetCount','likeCount','quoteCount']
# Only load wanted columns
elon_tweets = elon_tweets[column_list]

# Lowercase tweets
elon_tweets['content'] = elon_tweets['content'].apply(lambda x: x.lower())
# Tokenize 
elon_tweets['content'] = elon_tweets['content'].apply(lambda x: nltk.word_tokenize(x))
# Remove non-alphabetic
elon_tweets['content'] = elon_tweets['content'].apply(lambda x: [word for word in x if word.isalpha()])
# Count most common words
words_df = pd.DataFrame(Counter(itertools.chain(*elon_tweets['content'])).most_common(1000))
words_df.columns = [['word','word_count']]

# Add columns for collective tweet engagement parameter
elon_tweets['tweet_engagement'] = elon_tweets['likeCount'] + elon_tweets['quoteCount'] + elon_tweets['retweetCount']
# Normalize tweet engagement
elon_tweets['tweet_engagement'] = (elon_tweets['tweet_engagement'] - elon_tweets['tweet_engagement'].min()) / (elon_tweets['tweet_engagement'].max() - elon_tweets['tweet_engagement'].min())    
# Drop like, quote, and retweet count columns
elon_tweets = elon_tweets.drop(columns=['likeCount','quoteCount','retweetCount','replyCount'])
# Remake dates
elon_tweets['date'] = pd.to_datetime(elon_tweets['date'],format='%Y-%m-%d',utc=True).dt.date
# Get number of tweets per day
tweet_counts = elon_tweets.groupby('date').size().values
# Aggregate per date
elon_tweets = elon_tweets.groupby(elon_tweets['date']).sum(numeric_only=True)
# Add tweet counts to df
elon_tweets['tweet_counts'] = tweet_counts

for df in stock_dfs:
    df.columns = map(str.lower,df.columns)
    df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d',utc=True).dt.date
    df['close'] = df['close'].astype(float)
    df['close_normalized'] = (df['close'] - df['close'].min()) / (df['close'].max() - df['close'].min())

tesla = tesla_stock.rename(columns={'close':'tesla_close_price','close_normalized':'tesla_normalized'})[['date','tesla_close_price','tesla_normalized']]
twitter = twitter_stock.rename(columns={'close':'twitter_close_price','close_normalized':'twitter_normalized'})[['date','twitter_close_price','twitter_normalized']]
bitcoin = bitcoin.rename(columns={'close':'bitcoin_close_price','close_normalized':'bitcoin_normalized'})[['date','bitcoin_close_price','bitcoin_normalized']]
dogecoin = dogecoin.rename(columns={'close':'dogecoin_close_price','close_normalized':'dogecoin_normalized'})[['date','dogecoin_close_price','dogecoin_normalized']]

# Left join stock dataframes unto tweets
elon_tweets = elon_tweets.merge(tesla, on='date',how='left').merge(twitter, on='date',how='left').merge(bitcoin, on='date',how='left').merge(dogecoin, on='date',how='left')
# Check for and fill NaN values
nan_columns = elon_tweets.columns[elon_tweets.isna().any()].tolist()
print("Number of NaN values total: {0} and the columns that contain empty values: {1}".format(elon_tweets.isna().sum().sum(), nan_columns))
for col in elon_tweets.columns:
    if elon_tweets[col].dtype in ('int','float'):
        elon_tweets[col] = elon_tweets[col].fillna(0)

print("Number of NaN values total: {0} and the columns that contain empty values: {1}".format(elon_tweets.isna().sum().sum(), nan_columns))

# Save to csv files
elon_tweets.to_csv('./data/cleaned/elon_tweets.csv',index=False)
words_df.to_csv('./data/cleaned/word_count.csv',index=False)