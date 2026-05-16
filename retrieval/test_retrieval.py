import json
from retrieval import get_context

# Sample event — swap in a real one from events.json
event = {
    "event_id": "EVT_001",
    "title": "Will the Fed cut rates in June 2026?",
    "rules": "Resolves YES if the Federal Reserve announces a rate cut at the June 2026 FOMC meeting.",
    "market_stats": {
        "Yes": {"last_price": 0.62, "yes_ask": 0.63, "no_ask": 0.38},
        "No":  {"last_price": 0.38, "yes_ask": 0.37, "no_ask": 0.62}
    }
}

# --- P2 output (yours) ---
context = get_context(event["event_id"], event["title"], event["rules"])
print("\n========== P2 CONTEXT ==========")
print(json.dumps(context, indent=2))

# --- P3 output (theirs) ---
# Uncomment once P3 has their module ready
# from forecaster import get_prediction
# prediction = get_prediction(event, context)
# print("\n========== P3 PREDICTION ==========")
# print(json.dumps(prediction, indent=2))