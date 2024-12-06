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

# show categories list and count
print(news_categories['category'].value_counts())


# get 10% of the categories data
news_categories = news_categories.sample(frac=0.20, random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    news_categories['headline'], news_categories['category'], test_size=0.2, random_state=42
)

# Preprocess the text data and remove stop words
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

print(X_train_vectors.shape)

# Create the SVC model with specified hyperparameters
svc = svm.SVC(probability=True, C=10, class_weight='balanced', degree=2, gamma='scale', kernel='linear')

# Train the model
svc.fit(X_train_vectors, y_train)

# Predict on the test set
y_pred = svc.predict(X_test_vectors)

# Print the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Predict probabilities
y_proba = svc.predict_proba(X_test_vectors)

# Get top 2 predictions
import numpy as np
top_n = 2
top_n_pred = np.argsort(y_proba, axis=1)[:, -top_n:]
classes = svc.classes_
top_n_labels = classes[top_n_pred]

# Evaluate the model
y_test_array = y_test.to_numpy()
correct = [1 if y_test_array[i] in top_n_labels[i] else 0 for i in range(len(y_test))]
top_n_accuracy = np.mean(correct)
print(f"Top {top_n} accuracy: {top_n_accuracy}")

# Apply the model to news_data
news_data_vectors = vectorizer.transform(news_data['headline_text'])
news_proba = svc.predict_proba(news_data_vectors)
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
                precision    recall  f1-score   support

          ARTS       0.21      0.18      0.19        56
ARTS & CULTURE       0.23      0.13      0.17        52
  BLACK VOICES       0.40      0.31      0.35       193
      BUSINESS       0.31      0.42      0.36       245
       COLLEGE       0.41      0.33      0.37        48
        COMEDY       0.41      0.40      0.40       217
         CRIME       0.44      0.51      0.48       144
CULTURE & ARTS       0.33      0.07      0.11        46
       DIVORCE       0.64      0.73      0.68       132
     EDUCATION       0.32      0.40      0.36        47
 ENTERTAINMENT       0.48      0.62      0.54       681
   ENVIRONMENT       0.29      0.24      0.26        63
         FIFTY       0.18      0.13      0.15        68
  FOOD & DRINK       0.53      0.50      0.51       246
     GOOD NEWS       0.28      0.16      0.20        50
         GREEN       0.30      0.33      0.31        98
HEALTHY LIVING       0.18      0.22      0.20       254
 HOME & LIVING       0.65      0.62      0.63       172
        IMPACT       0.23      0.21      0.22       125
 LATINO VOICES       0.62      0.21      0.31        48
         MEDIA       0.56      0.41      0.47       135
         MONEY       0.46      0.36      0.41        74
     PARENTING       0.42      0.51      0.46       341
       PARENTS       0.24      0.19      0.21       161
      POLITICS       0.66      0.73      0.69      1433
  QUEER VOICES       0.63      0.53      0.57       251
      RELIGION       0.59      0.41      0.48        95
       SCIENCE       0.54      0.34      0.42        93
        SPORTS       0.63      0.58      0.61       210
         STYLE       0.35      0.24      0.28        96
STYLE & BEAUTY       0.68      0.66      0.67       378
         TASTE       0.13      0.10      0.11        82
          TECH       0.37      0.37      0.37        70
 THE WORLDPOST       0.38      0.32      0.34       142
        TRAVEL       0.62      0.64      0.63       389
     U.S. NEWS       0.20      0.04      0.06        54
      WEDDINGS       0.71      0.66      0.69       149
    WEIRD NEWS       0.26      0.16      0.20       111
      WELLNESS       0.45      0.51      0.48       755
         WOMEN       0.27      0.30      0.28       138
    WORLD NEWS       0.32      0.23      0.26       133
     WORLDPOST       0.29      0.17      0.21       106

      accuracy                           0.49      8381
     macro avg       0.41      0.36      0.37      8381
  weighted avg       0.49      0.49      0.48      8381

Top 2 accuracy: 0.660064431452094
'''