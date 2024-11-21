import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import os

# Load datasets
fake_df = pd.read_csv(r'E:\Sem V\ML\fake-news-detection\data\Fake.csv')
true_df = pd.read_csv(r'E:\Sem V\ML\fake-news-detection\data\True.csv')

# Combine datasets and create labels
fake_df['label'] = 0  # 0 for fake
true_df['label'] = 1  # 1 for real
data = pd.concat([fake_df, true_df], ignore_index=True)

# Split data into features and labels
X = data['text']  # Use the 'text' column for predictions
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes model
nb_model = MultinomialNB()
print("Training Naive Bayes model...")
nb_model.fit(X_train_tfidf, y_train)

# Define paths for saving the model and vectorizer
models_dir = os.path.join(os.path.dirname(__file__), '../models')

# Save the model
with open(os.path.join(models_dir, 'naive_bayes.pkl'), 'wb') as f:
    pickle.dump(nb_model, f)

# Save the vectorizer
with open(os.path.join(models_dir, 'vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

# Evaluate model
y_pred = nb_model.predict(X_test_tfidf)
print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred))

print("Naive Bayes model trained and saved successfully.")
