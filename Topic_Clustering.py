import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load news data
news_data = pd.read_csv("abcnews-date-text.csv", names=['date', 'headline_text'])

# Convert date column to datetime
news_data['date'] = pd.to_datetime(news_data['date'], format='%Y%m%d')

# Load stock data
stock_data = pd.read_csv("ASX-200-Historical-Data.csv", header=0)

# Rename columns to match code expectations
stock_data.columns = ['date', 'price', 'open', 'high', 'low', 'vol', 'change_pct']

# Remove commas from numbers and convert to float
numeric_cols = ['price', 'open', 'high', 'low']
for col in numeric_cols:
    stock_data[col] = stock_data[col].str.replace(',', '').astype(float)

# Remove percent sign and convert 'change_pct' to float
stock_data['change_pct'] = stock_data['change_pct'].str.replace('%', '').astype(float)

# Convert 'vol' to numeric (handle 'M' for millions and 'K' for thousands)
def parse_volume(value):
    if isinstance(value, str):
        value = value.replace(',', '')
        if 'B' in value:
            return float(value.replace('B', '')) * 1e9
        elif 'M' in value:
            return float(value.replace('M', '')) * 1e6
        elif 'K' in value:
            return float(value.replace('K', '')) * 1e3
        else:
            return float(value)
    else:
        return value

stock_data['vol'] = stock_data['vol'].apply(parse_volume)

# Convert date columns to datetime using the correct format
stock_data['date'] = pd.to_datetime(stock_data['date'], format='%m/%d/%Y')

# Sort the stock data by date
stock_data = stock_data.sort_values('date')

# Compute daily returns
stock_data['return'] = stock_data['price'].pct_change()

# Merge news and stock data on date
merged_data = pd.merge(news_data, stock_data, on='date')

# Group headlines by date
grouped_headlines = merged_data.groupby('date')['headline_text'].apply(' '.join).reset_index()

# Align returns with headlines
returns = merged_data[['date', 'return']].drop_duplicates()
returns = returns[returns['date'].isin(grouped_headlines['date'])]
y = returns['return'].values

# Remove any NaN returns
valid_indices = ~np.isnan(y)
X_text = grouped_headlines.loc[valid_indices, 'headline_text']
y = y[valid_indices]

# Vectorize the headlines
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X_text)

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Get feature names and coefficients
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_

# Create a DataFrame of words and their coefficients
word_impact = pd.DataFrame({
    'word': feature_names,
    'coefficient': coefficients
})

# Sort words by absolute value of coefficients
word_impact['abs_coefficient'] = word_impact['coefficient'].abs()
word_impact = word_impact.sort_values('abs_coefficient', ascending=False)

# Display top words impacting the stock price
print(word_impact[['word', 'coefficient']].head(50))
