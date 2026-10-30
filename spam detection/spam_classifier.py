import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
nltk.download("stopwords")

# dataset

df = pd.read_csv("spam.csv", encoding="latin-1")


df = df[["v1", "v2"]]
df.columns = ["label", "message"]

df["label"] = df["label"].map({"ham": 0, "spam": 1})

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["clean_message"] = df["message"].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["clean_message"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n========== MODEL PERFORMANCE ==========\n")

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%\n")

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved successfully!")

while True:

    message = input("\nEnter a message (or type 'exit' to quit): ")

    if message.lower() == "exit":
        break

    cleaned = clean_text(message)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        print("\nPrediction: SPAM")
    else:
        print("\nPrediction: NOT SPAM")