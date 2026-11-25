
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("data/news.csv")

# Separate input and output
X = data["text"]
y = data["label"]

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train AI model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Training Completed!")
print("Accuracy:", accuracy)
print(classification_report(y_test, predictions))

# Save model
with open("fake_news_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Model and Vectorizer Saved Successfully!")
