import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from typesafe_sdk import Choice, TypeSafeAuthenticationError, TypeSafeClient, TypeSafeError

MODEL = "jev-1.13.0"  # pinned, so the results don't change if the "latest" alias moves

# USD per million input tokens. Output tokens are free.
# Check the current price with TypeSafe before you publish.
INPUT_PRICE_PER_MTOK = 0.042

# Same definitions as the Claude Haiku prompt, word for word.
QUESTIONS = {
    "intent": Choice(
        instructions=(
            "Choose the one intent that best describes what the customer wants in `message`. "
            "If the message contains more than one request, choose the first one the customer asks for."
        ),
        criteria={
            "price_availability": "asks about prices, the menu, what is available, opening hours, delivery fees or portion sizes, before ordering.",
            "place_order": "wants to order, add to an order or change an order.",
            "payment_confirmation": "says they have paid and wants the payment confirmed.",
            "order_status": "asks where their order is, when it will arrive or whether it is ready.",
            "complaint_refund": "reports a problem with an order (wrong, missing, cold or late food, rude service) or asks for a refund.",
            "other": "fits none of the above, such as greetings, job enquiries or supplier pitches.",
        },
    )
}


def classify(client, message):
    start = time.perf_counter()
    try:
        response = client.system_one(state={"message": message}, questions=QUESTIONS)
    except TypeSafeAuthenticationError:
        raise
    except TypeSafeError:
        return "error", 0.0, time.perf_counter() - start, 0
    latency = time.perf_counter() - start
    answer = response.choices["intent"]
    return answer.choice, answer.confidence, latency, response.usage.input_tokens or 0


def main(split):
    df = pd.read_csv("data/messages.csv")
    data = df[df["split"] == split].copy()

    with TypeSafeClient(model=MODEL) as client:  # reads TYPESAFE_API_KEY from your environment
        results = [classify(client, m) for m in data["message"]]

    data["predicted"] = [r[0] for r in results]
    data["confidence"] = [r[1] for r in results]
    data["latency_s"] = [r[2] for r in results]
    data["input_tokens"] = [r[3] for r in results]

    # The stress set labels messages that fit no intent as "out_of_scope".
    data["predicted"] = data["predicted"].replace({"other": "out_of_scope"})

    os.makedirs("results", exist_ok=True)
    data.to_csv(f"results/jev_{split}.csv", index=False)

    print(f"Messages: {len(data)}")
    print(f"Accuracy: {accuracy_score(data['intent'], data['predicted']):.3f}")
    if split == "stress_multi":
        lenient = data.apply(
            lambda r: r["predicted"] == r["intent"]
            or (isinstance(r["note"], str) and r["predicted"] in r["note"]),
            axis=1,
        )
        print(f"Accuracy if the second intent also counts: {lenient.mean():.3f}")
    print(classification_report(data["intent"], data["predicted"], zero_division=0))

    right = data["predicted"] == data["intent"]
    print(f"Mean confidence when right: {data.loc[right, 'confidence'].mean():.2f}")
    if (~right).any():
        print(f"Mean confidence when wrong: {data.loc[~right, 'confidence'].mean():.2f}")

    lat = data["latency_s"]
    print(f"Latency (s): mean {lat.mean():.2f}, median {lat.median():.2f}, p95 {np.percentile(lat, 95):.2f}")

    tin = data["input_tokens"].sum()
    cost = tin * INPUT_PRICE_PER_MTOK / 1e6
    print(f"Tokens: {tin} input")
    print(f"Cost: ${cost:.5f} total, ${cost / len(data) * 1000:.4f} per 1,000 messages")
    print(f"Invalid or failed calls: {(data['predicted'] == 'error').sum()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
