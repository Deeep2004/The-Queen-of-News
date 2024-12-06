import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the data
news_data = pd.read_csv("5%_abcnews-date-text.csv", names=['date', 'headline_text'])
news_categories = pd.read_json("News_Category_Dataset_v3.json", lines=True)
news_categories = news_categories[['category', 'headline']]

# show categories list and count
print(news_categories['category'].value_counts())

# Preprocess the data
X = news_categories['headline']
y = news_categories['category']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

# Train the Logistic Regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tf, y_train)

# Make predictions
y_pred = clf.predict(X_test_tf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict probabilities
y_proba = clf.predict_proba(X_test_tf)

# Get top 2 predictions
import numpy as np
top_n = 2
top_n_pred = np.argsort(y_proba, axis=1)[:, -top_n:]
classes = clf.classes_
top_n_labels = classes[top_n_pred]

# Evaluate the model
y_test_array = y_test.to_numpy()
correct = [1 if y_test_array[i] in top_n_labels[i] else 0 for i in range(len(y_test))]
top_n_accuracy = np.mean(correct)
print(f"Top {top_n} accuracy: {top_n_accuracy}")

# Calculate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Apply the model to news_data
news_data_vectors = vectorizer.transform(news_data['headline_text'])
news_proba = clf.predict_proba(news_data_vectors)
top_n_pred_news = np.argsort(news_proba, axis=1)[:, -top_n:]
top_n_labels_news = classes[top_n_pred_news]

# Assign the top 2 predicted categories
news_data['predicted_category_1'] = top_n_labels_news[:, 0]
news_data['predicted_category_2'] = top_n_labels_news[:, 1]

# Display predictions
print(news_data[['headline_text', 'predicted_category_1', 'predicted_category_2']].head())

# Save the predictions to a CSV file
news_data.to_csv('predictions.csv', index=False)