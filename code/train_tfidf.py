import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

df = pd.read_csv("data/messages.csv")
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

model = make_pipeline(
    TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True),
    LogisticRegression(max_iter=1000),
)

start = time.perf_counter()
model.fit(train["message"], train["intent"])
print(f"Training time: {time.perf_counter() - start:.3f} seconds")

preds = model.predict(test["message"])
print(f"Test accuracy: {accuracy_score(test['intent'], preds):.3f}")
print(classification_report(test["intent"], preds))

# Messages the model got wrong
wrong = test.assign(predicted=preds)
wrong = wrong[wrong["intent"] != wrong["predicted"]]
print(wrong[["message", "intent", "predicted"]].to_string(index=False))

# Time a single prediction
start = time.perf_counter()
model.predict(["I've paid 12k, please confirm"])
print(f"Single prediction: {(time.perf_counter() - start) * 1000:.2f} ms")

joblib.dump(model, "tfidf_model.joblib")
