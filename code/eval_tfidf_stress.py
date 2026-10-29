import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

df = pd.read_csv("data/messages.csv")
train = df[df["split"] == "train"]

# Same model and settings as before
model = make_pipeline(
    TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True),
    LogisticRegression(max_iter=1000),
)
model.fit(train["message"], train["intent"])


def run(split):
    data = df[df["split"] == split].copy()
    proba = model.predict_proba(data["message"])
    data["predicted"] = model.classes_[proba.argmax(axis=1)]
    data["confidence"] = proba.max(axis=1)
    return data


os.makedirs("results", exist_ok=True)

# 1. Pidgin and typos
pidgin = run("stress_pidgin")
correct = (pidgin["predicted"] == pidgin["intent"]).sum()
print(f"Pidgin and typos: {correct}/{len(pidgin)} correct ({correct / len(pidgin):.0%})")
print(pidgin[pidgin["predicted"] != pidgin["intent"]][["message", "intent", "predicted"]].to_string(index=False))

# 2. Messages that ask for two things
multi = run("stress_multi")
strict = (multi["predicted"] == multi["intent"]).sum()
lenient = multi.apply(
    lambda r: r["predicted"] == r["intent"]
    or (isinstance(r["note"], str) and r["predicted"] in r["note"]),
    axis=1,
).sum()
print(f"\nTwo-intent messages: {strict}/{len(multi)} match the first intent")
print(f"Two-intent messages: {lenient}/{len(multi)} match either intent")

# 3. Messages that fit none of the intents
test = run("test")
oos = run("stress_oos")
print("\nOut-of-scope messages: the model has no 'other' class, so none can be right.")
print(oos["predicted"].value_counts().to_string())

# A confidence cutoff: reject anything below the 10th percentile of confidence on the test set
cutoff = np.percentile(test["confidence"], 10)
print(f"\nMedian confidence, in-scope test messages: {test['confidence'].median():.2f}")
print(f"Median confidence, out-of-scope messages:  {oos['confidence'].median():.2f}")
print(f"Cutoff (10th percentile of test confidence): {cutoff:.3f}")
print(f"Out-of-scope messages rejected: {(oos['confidence'] < cutoff).sum()}/{len(oos)}")
print(f"In-scope test messages wrongly rejected: {(test['confidence'] < cutoff).sum()}/{len(test)}")

pd.concat([pidgin, multi, oos]).to_csv("results/tfidf_stress.csv", index=False)
