import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

# Load the model, tokenizer, and label encoder
model = tf.keras.models.load_model('bilstm_model.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as handle:  # We need to save this during training
    label_encoder = pickle.load(handle)

# Load ABC News data
news_data = pd.read_csv("abcnews-date-text.csv", names=['date', 'headline_text'])

# Sample 5% of the data
news_data = news_data.sample(frac=0.05, random_state=42)

# Tokenize and pad all headlines
max_len = 50
sequences = tokenizer.texts_to_sequences(news_data['headline_text'])
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

# Get predictions
predictions = model.predict(padded_sequences)
top_2_indices = np.argsort(predictions, axis=1)[:, -2:]

# Convert indices to category names
news_data['category_1'] = label_encoder.inverse_transform(top_2_indices[:, 1])
news_data['category_2'] = label_encoder.inverse_transform(top_2_indices[:, 0])

# Save predictions
news_data.to_csv('abc_news_with_predictions.csv', index=False)

print("First few predictions:")
print(news_data[['headline_text', 'category_1', 'category_2']].head())