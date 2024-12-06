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

# get 10% of the categories data
news_categories = news_categories.sample(frac=0.1, random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    news_categories['headline'], news_categories['category'], test_size=0.2, random_state=42
)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

print(X_train_vectors.shape)

# Define the parameter grid
param_grid = {
    'C': [1, 10],
    'gamma': ['scale'],
    'kernel': ['rbf', 'linear', 'poly'],
    'degree': [2, 3, 4],
    'class_weight': ['balanced']
}
# Create the SVM classifier with probability=True
svc = svm.SVC(probability=True)

# Perform grid search
grid_search = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_vectors, y_train)

# Best model
best_svc = grid_search.best_estimator_

# Print the best parameters used
print("Best parameters:", grid_search.best_params_)

# Predict on the test set
y_pred = best_svc.predict(X_test_vectors)

# Print the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

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
'''
Best parameters: {'C': 10, 'class_weight': 'balanced', 'degree': 2, 'gamma': 'scale', 'kernel': 'linear'}
                precision    recall  f1-score   support

          ARTS       0.08      0.03      0.05        29
ARTS & CULTURE       0.14      0.09      0.11        22
  BLACK VOICES       0.30      0.26      0.28       103
      BUSINESS       0.29      0.34      0.31       122
       COLLEGE       0.00      0.00      0.00        14
        COMEDY       0.30      0.29      0.30       100
         CRIME       0.44      0.53      0.48        68
CULTURE & ARTS       0.83      0.24      0.37        21
       DIVORCE       0.62      0.52      0.56        62
     EDUCATION       0.29      0.24      0.26        17
 ENTERTAINMENT       0.44      0.63      0.52       358
   ENVIRONMENT       0.62      0.17      0.27        29
         FIFTY       0.25      0.12      0.17        32
  FOOD & DRINK       0.62      0.52      0.56       128
     GOOD NEWS       0.23      0.10      0.14        31
         GREEN       0.24      0.25      0.25        55
HEALTHY LIVING       0.19      0.27      0.22       112
 HOME & LIVING       0.64      0.59      0.62        98
        IMPACT       0.20      0.11      0.14        73
 LATINO VOICES       0.57      0.19      0.29        21
         MEDIA       0.59      0.37      0.45        60
         MONEY       0.22      0.11      0.14        47
     PARENTING       0.40      0.48      0.43       168
       PARENTS       0.24      0.16      0.19        86
      POLITICS       0.61      0.74      0.67       692
  QUEER VOICES       0.62      0.53      0.57       133
      RELIGION       0.53      0.37      0.43        54
       SCIENCE       0.61      0.27      0.37        52
        SPORTS       0.54      0.45      0.49        95
         STYLE       0.43      0.20      0.27        50
STYLE & BEAUTY       0.65      0.71      0.68       206
         TASTE       0.33      0.14      0.20        43
          TECH       0.48      0.32      0.39        37
 THE WORLDPOST       0.48      0.38      0.43        78
        TRAVEL       0.60      0.58      0.59       195
     U.S. NEWS       0.00      0.00      0.00        31
      WEDDINGS       0.79      0.69      0.74        61
    WEIRD NEWS       0.43      0.24      0.31        49
      WELLNESS       0.44      0.56      0.49       368
         WOMEN       0.33      0.28      0.30        68
    WORLD NEWS       0.28      0.21      0.24        63
     WORLDPOST       0.33      0.18      0.24        60

      accuracy                           0.48      4191
     macro avg       0.41      0.32      0.35      4191
  weighted avg       0.47      0.48      0.46      4191

Top 2 accuracy: 0.6323073252207111
                                       headline_text predicted_category_1 predicted_category_2
0  aba decides against community broadcasting lic...               IMPACT             POLITICS
1                    sunday november 11 full program             POLITICS             WELLNESS
2    tas man sentenced in us court over baby battery            GOOD NEWS                CRIME
3                       mixed response to water plan                GREEN             POLITICS
4              party to decide future of clp senator             WEDDINGS             POLITICS
'''