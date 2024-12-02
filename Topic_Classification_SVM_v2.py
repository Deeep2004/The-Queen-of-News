import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load the data
news_data = pd.read_csv("5%_abcnews-date-text.csv", names=['date', 'headline_text'])
news_categories = pd.read_json("News_Category_Dataset_v3.json", lines=True)
news_categories = news_categories[['category', 'headline']]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    news_categories['headline'], news_categories['category'], test_size=0.2, random_state=42
)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}

# Create the SVM classifier with probability=True
svc = svm.SVC(probability=True)

# Perform grid search
grid_search = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_vectors, y_train)

# Best model
best_svc = grid_search.best_estimator_

# Predict probabilities
y_proba = best_svc.predict_proba(X_test_vectors)

# Get top 2 predictions
import numpy as np
top_n = 2
top_n_pred = np.argsort(y_proba, axis=1)[:, -top_n:]
classes = best_svc.classes_
top_n_labels = classes[top_n_pred]

# Evaluate the model
y_test_array = y_test.to_numpy()
correct = [1 if y_test_array[i] in top_n_labels[i] else 0 for i in range(len(y_test))]
top_n_accuracy = np.mean(correct)
print(f"Top {top_n} accuracy: {top_n_accuracy}")

# Apply the model to news_data
news_data_vectors = vectorizer.transform(news_data['headline_text'])
news_proba = best_svc.predict_proba(news_data_vectors)
top_n_pred_news = np.argsort(news_proba, axis=1)[:, -top_n:]
top_n_labels_news = classes[top_n_pred_news]

# Assign the top 2 predicted categories
news_data['predicted_category_1'] = top_n_labels_news[:, 0]
news_data['predicted_category_2'] = top_n_labels_news[:, 1]

# Display predictions
print(news_data[['headline_text', 'predicted_category_1', 'predicted_category_2']].head())

# Save the predictions to a CSV file
news_data.to_csv('predictions.csv', index=False)
