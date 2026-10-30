import os
import sys
import time

import anthropic
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

MODEL = "claude-haiku-5-5"

# USD per million tokens, for prompts up to 100,000 tokens.
INPUT_PRICE_PER_MTOK = 0.10
OUTPUT_PRICE_PER_MTOK = 0.50

INTENTS = [
    "price_availability",
    "place_order",
    "payment_confirmation",
    "order_status",
    "complaint_refund",
    "other",
]

SYSTEM_PROMPT = """You classify customer messages sent to a food business on WhatsApp and Instagram.
Choose the one intent that best describes what the customer wants.

Intents:
- price_availability: asks about prices, the menu, what is available, opening hours, delivery fees or portion sizes, before ordering.
- place_order: wants to order, add to an order or change an order.
- payment_confirmation: says they have paid and wants the payment confirmed.
- order_status: asks where their order is, when it will arrive or whether it is ready.
- complaint_refund: reports a problem with an order (wrong, missing, cold or late food, rude service) or asks for a refund.
- other: fits none of the above, such as greetings, job enquiries or supplier pitches.

If a message contains more than one request, choose the first one the customer asks for."""

TOOLS = [
    {
        "name": "record_intent",
        "description": "Record the intent of the customer message.",
        "input_schema": {
            "type": "object",
            "properties": {"intent": {"type": "string", "enum": INTENTS}},
            "required": ["intent"],
        },
    }
]


def classify(client, message):
    start = time.perf_counter()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=100,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            tool_choice={"type": "tool", "name": "record_intent"},
            messages=[{"role": "user", "content": message}],
        )
    except anthropic.APIError:
        return "error", time.perf_counter() - start, 0, 0
    latency = time.perf_counter() - start

    intent = "invalid"
    for block in response.content:
        if block.type == "tool_use" and block.input.get("intent") in INTENTS:
            intent = block.input["intent"]
    return intent, latency, response.usage.input_tokens, response.usage.output_tokens


def main(split):
    df = pd.read_csv("data/messages.csv")
    data = df[df["split"] == split].copy()
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from your environment

    results = [classify(client, m) for m in data["message"]]
    data["predicted"] = [r[0] for r in results]
    data["latency_s"] = [r[1] for r in results]
    data["input_tokens"] = [r[2] for r in results]
    data["output_tokens"] = [r[3] for r in results]

    # The stress set labels messages that fit no intent as "out_of_scope".
    data["predicted"] = data["predicted"].replace({"other": "out_of_scope"})

    os.makedirs("results", exist_ok=True)
    data.to_csv(f"results/llm_{split}.csv", index=False)

    print(f"Messages: {len(data)}")
    print(f"Accuracy: {accuracy_score(data['intent'], data['predicted']):.3f}")
    if split == "stress_multi":
        # Also count the answer as right if it matches the second intent in the note.
        lenient = data.apply(
            lambda r: r["predicted"] == r["intent"]
            or (isinstance(r["note"], str) and r["predicted"] in r["note"]),
            axis=1,
        )
        print(f"Accuracy if the second intent also counts: {lenient.mean():.3f}")
    print(classification_report(data["intent"], data["predicted"], zero_division=0))

    lat = data["latency_s"]
    print(f"Latency (s): mean {lat.mean():.2f}, median {lat.median():.2f}, p95 {np.percentile(lat, 95):.2f}")

    tin, tout = data["input_tokens"].sum(), data["output_tokens"].sum()
    cost = tin * INPUT_PRICE_PER_MTOK / 1e6 + tout * OUTPUT_PRICE_PER_MTOK / 1e6
    print(f"Tokens: {tin} input, {tout} output")
    print(f"Cost: ${cost:.4f} total, ${cost / len(data) * 1000:.4f} per 1,000 messages")
    print(f"Invalid or failed calls: {(data['predicted'].isin(['invalid', 'error'])).sum()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")