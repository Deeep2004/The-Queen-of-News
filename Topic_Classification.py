import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Import data
news_data = pd.read_csv("5%_abcnews-date-text.csv", names=['date', 'headline_text'])
news_categories = pd.read_json("News_Category_Dataset_v3.json", lines=True)
news_categories = news_categories[['category', 'headline']]

# Encode labels
label_encoder = LabelEncoder()
news_categories['label'] = label_encoder.fit_transform(news_categories['category'])

# Split the category dataset
X = news_categories['headline']
y = news_categories['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_encoder.classes_)
)

# Tokenize data
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

# Convert labels to tensors
y_train = tf.convert_to_tensor(y_train.values, dtype=tf.int64)
y_test = tf.convert_to_tensor(y_test.values, dtype=tf.int64)

# Prepare TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
)).shuffle(len(y_train)).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
)).batch(16)

# Compile the model with the appropriate loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=3, validation_data=test_dataset)

# Evaluate the model
y_pred = model.predict(test_dataset).logits
y_pred_labels = tf.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_))

# Predict categories for news_data
news_headlines = news_data['headline_text']
news_encodings = tokenizer(
    list(news_headlines),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors='tf'
)
news_dataset = tf.data.Dataset.from_tensor_slices(dict(news_encodings)).batch(16)
news_predictions = model.predict(news_dataset).logits
news_pred_labels = tf.argmax(news_predictions, axis=1)
news_data['category'] = label_encoder.inverse_transform(news_pred_labels)

# Save the labeled dataset
news_data.to_csv("abcnews_with_categories.csv", index=False)