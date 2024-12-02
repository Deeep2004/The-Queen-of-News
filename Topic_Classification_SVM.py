import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score

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

# Train the SVM classifier
clf = svm.SVC()
clf.fit(X_train_vectors, y_train)

# Evaluate the model
y_pred = clf.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Apply the model to news_data
news_data_vectors = vectorizer.transform(news_data['headline_text'])
news_data['predicted_category'] = clf.predict(news_data_vectors)

# Display predictions
print(news_data[['headline_text', 'predicted_category']].head())

# Save the predictions to a CSV file
news_data.to_csv('predictions.csv', index=False)
