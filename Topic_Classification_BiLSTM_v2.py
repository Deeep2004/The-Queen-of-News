import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Define category mapping
category_mapping = {
    'LIFESTYLE AND WELLNESS': ['WELLNESS', 'HEALTHY LIVING', 'HOME & LIVING', 'STYLE & BEAUTY', 'STYLE'],
    'PARENTING AND EDUCATION': ['PARENTING', 'PARENTS', 'EDUCATION', 'COLLEGE'],
    'SPORTS AND ENTERTAINMENT': ['SPORTS', 'ENTERTAINMENT', 'COMEDY', 'WEIRD NEWS', 'ARTS'],
    'TRAVEL-TOURISM & ART-CULTURE': ['TRAVEL', 'ARTS & CULTURE', 'CULTURE & ARTS', 'FOOD & DRINK', 'TASTE'],
    'EMPOWERED VOICES': ['WOMEN', 'QUEER VOICES', 'LATINO VOICES', 'BLACK VOICES'],
    'BUSINESS-MONEY': ['BUSINESS', 'MONEY'],
    'WORLDNEWS': ['THE WORLDPOST', 'WORLDPOST', 'WORLD NEWS'],
    'ENVIRONMENT': ['ENVIRONMENT', 'GREEN'],
    'SCIENCE AND TECH': ['TECH', 'SCIENCE'],
    'GENERAL': ['FIFTY', 'IMPACT', 'GOOD NEWS', 'CRIME'],
    'MISC': ['WEDDINGS', 'DIVORCE', 'RELIGION', 'MEDIA']
}

# Load the data
news_categories = pd.read_json("News_Category_Dataset_v3.json", lines=True)
news_categories = news_categories[['category', 'headline']]

# Function to map categories to groups
def map_category_to_group(category):
    for group, categories in category_mapping.items():
        if category.upper() in [cat.upper() for cat in categories]:
            return group
    return 'OTHER'

# Apply the grouping
news_categories['category'] = news_categories['category'].apply(map_category_to_group)

# use 1% of the data for testing
# news_categories = news_categories.sample(frac=0.01, random_state=42)

# Continue with the rest of your existing code
X = news_categories['headline']
y = news_categories['category']

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Tokenize the text
max_words = 20000  # Maximum number of words to keep
max_len = 50  # Maximum length of each sequence

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Build the BiLSTM model
model = Sequential([
    Embedding(max_words, 100, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss',
                              patience=3,
                              restore_best_weights=True)

# Train the model
history = model.fit(X_train_pad, y_train,
                   epochs=10,
                   batch_size=64,
                   validation_split=0.2,
                   callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f'Test accuracy: {accuracy:.4f}')
# Get predictions
y_pred_proba = model.predict(X_test_pad)

# Get top 2 predictions indices
top_2_pred = np.argsort(y_pred_proba, axis=1)[:, -2:]

# Convert predictions to class labels - handling each column separately
top_2_classes = np.column_stack([
    label_encoder.inverse_transform(top_2_pred[:, 0]),
    label_encoder.inverse_transform(top_2_pred[:, 1])
])

# Calculate top-2 accuracy
y_true = label_encoder.inverse_transform(np.argmax(y_test, axis=1))
top_2_accuracy = sum([y_true[i] in top_2_classes[i] for i in range(len(y_true))]) / len(y_true)
print(f'Top 2 accuracy: {top_2_accuracy:.4f}')
# Save the model and tokenizer for later use
# Save the model using the modern Keras format
model.save('bilstm_model.keras')
import pickle

with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle)