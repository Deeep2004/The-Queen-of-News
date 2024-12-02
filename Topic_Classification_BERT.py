import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizerFast, TFBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.optimizers import legacy

# Load the data
news_categories = pd.read_json("News_Category_Dataset_v3.json", lines=True)
news_categories = news_categories[['category', 'headline']]

# Encode labels
le = LabelEncoder()
news_categories['category_encoded'] = le.fit_transform(news_categories['category'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    news_categories['headline'],
    news_categories['category_encoded'],
    test_size=0.2,
    random_state=42
)

# Initialize tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(le.classes_)
)

# Tokenize the data
train_encodings = tokenizer(
    list(X_train),
    truncation=True,
    padding=True,
    max_length=128
)
test_encodings = tokenizer(
    list(X_test),
    truncation=True,
    padding=True,
    max_length=128
)

# Prepare datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    list(y_train)
)).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    list(y_test)
)).batch(16)

# Compile the model with the legacy optimizer
optimizer = legacy.Adam(learning_rate=2e-5)
model.compile(
    optimizer=optimizer,
    loss=model.compute_loss,
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_dataset,
    epochs=3,
    validation_data=test_dataset
)

# Evaluate the model
loss, accuracy = model.evaluate(test_dataset)
print(f"Model accuracy: {accuracy}")

# Apply the model to news_data
news_data = pd.read_csv("5%_abcnews-date-text.csv", names=['date', 'headline_text'])
news_encodings = tokenizer(
    list(news_data['headline_text']),
    truncation=True,
    padding=True,
    max_length=128
)
news_dataset = tf.data.Dataset.from_tensor_slices(dict(news_encodings)).batch(16)
predictions = model.predict(news_dataset)
predicted_labels = tf.argmax(predictions.logits, axis=1)
news_data['predicted_category'] = le.inverse_transform(predicted_labels)

# Save the predictions to a CSV file
news_data.to_csv('predictions.csv', index=False)