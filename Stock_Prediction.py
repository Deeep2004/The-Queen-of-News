import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
import xgboost as xgb
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load news data
logger.info("Loading news data...")
news_data = pd.read_csv("abcnews-date-text.csv", names=['date', 'headline_text'])

# Convert date column to datetime
logger.info("Converting date column to datetime...")
news_data['date'] = pd.to_datetime(news_data['date'], format='%Y%m%d')

# Load stock data
logger.info("Loading stock data...")
stock_data = pd.read_csv("ASX-200-Historical-Data.csv", header=0)

# Rename columns to match code expectations
logger.info("Renaming stock data columns...")
stock_data.columns = ['date', 'price', 'open', 'high', 'low', 'vol', 'change_pct']

# Remove commas from numbers and convert to float
logger.info("Cleaning numeric columns in stock data...")
numeric_cols = ['price', 'open', 'high', 'low']
for col in numeric_cols:
    stock_data[col] = stock_data[col].str.replace(',', '').astype(float)

# Remove percent sign and convert 'change_pct' to float
stock_data['change_pct'] = stock_data['change_pct'].str.replace('%', '').astype(float)

# Convert 'vol' to numeric (handle 'B' for billions, 'M' for millions, and 'K' for thousands)
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

logger.info("Parsing volume column in stock data...")
stock_data['vol'] = stock_data['vol'].apply(parse_volume)

# Convert date columns to datetime using the correct format
logger.info("Converting date columns to datetime...")
stock_data['date'] = pd.to_datetime(stock_data['date'], format='%m/%d/%Y')

# Sort the stock data by date
logger.info("Sorting stock data by date...")
stock_data = stock_data.sort_values('date')

# Compute daily returns
logger.info("Computing daily returns...")
stock_data['return'] = stock_data['price'].pct_change()

# Create labels for stock movement (1 for up, 0 for neutral, -1 for down)
def classify_movement(return_value):
    if return_value > 0.002:
        return 1
    elif return_value < -0.002:
        return -1
    else:
        return 0

logger.info("Classifying stock movements...")
stock_data['movement'] = stock_data['return'].apply(classify_movement)

# Map the labels to non-negative integers
label_mapping = {-1: 0, 0: 1, 1: 2}
stock_data['movement_mapped'] = stock_data['movement'].map(label_mapping)

# Merge news and stock data on date
logger.info("Merging news and stock data...")
merged_data = pd.merge(news_data, stock_data, on='date')

# Group headlines by date
logger.info("Grouping headlines by date...")
grouped_headlines = merged_data.groupby('date')['headline_text'].apply(' '.join).reset_index()

# Add sentiment analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

logger.info("Adding sentiment analysis...")
grouped_headlines['sentiment'] = grouped_headlines['headline_text'].apply(get_sentiment)

# Align returns with headlines
logger.info("Aligning returns with headlines...")
returns = merged_data[['date', 'movement_mapped']].drop_duplicates()
returns = returns[returns['date'].isin(grouped_headlines['date'])]
y = returns['movement_mapped'].values

# Remove any NaN returns
logger.info("Removing NaN returns...")
valid_indices = ~np.isnan(y)
X_text = grouped_headlines.loc[valid_indices, 'headline_text']
X_sentiment = grouped_headlines.loc[valid_indices, 'sentiment']
y = y[valid_indices]

# Vectorize the headlines
logger.info("Vectorizing headlines...")
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(X_text)

# Combine TF-IDF features with sentiment
logger.info("Combining TF-IDF features with sentiment...")
X = np.hstack((X_tfidf.toarray(), X_sentiment.values.reshape(-1, 1)))

# Split the data into training and testing sets
logger.info("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# Initialize the XGBoost model
logger.info("Initializing XGBoost model...")
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')

# Initialize Grid Search
logger.info("Initializing Grid Search...")
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit Grid Search to the training data
logger.info("Fitting Grid Search to the training data...")
sys.stdout.flush()  # Ensure all previous logs are printed
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
logger.info("Best parameters found: %s", grid_search.best_params_)
logger.info("Best accuracy score: %s", grid_search.best_score_)

# Use the best estimator to make predictions on the test set
logger.info("Making predictions on the test set...")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Reverse the label mapping for predictions
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
y_test_original = np.vectorize(reverse_label_mapping.get)(y_test)
y_pred_original = np.vectorize(reverse_label_mapping.get)(y_pred)

# Evaluate the model
logger.info("Evaluating the model...")
print("Accuracy:", accuracy_score(y_test_original, y_pred_original))
print(classification_report(y_test_original, y_pred_original))

# Function to predict stock movement based on new headlines
def predict_stock_movement(headlines):
    sentiment = get_sentiment(headlines)
    headlines_vectorized = vectorizer.transform([headlines])
    features = np.hstack((headlines_vectorized.toarray(), np.array([[sentiment]])))
    prediction = best_model.predict(features)
    original_prediction = reverse_label_mapping[prediction[0]]
    print(f"Prediction: {original_prediction}")
    if original_prediction == 1:
        return "Up"
    elif original_prediction == -1:
        return "Down"
    else:
        return "Neutral"

# Example usage
if __name__ == "__main__":
    while True:
        new_headlines = input("Enter headlines (or type 'exit' to quit): ")
        if new_headlines.lower() == 'exit':
            break
        print("Prediction for new headlines:", predict_stock_movement(new_headlines))
