import json

# validate prediction "Yes" and "No" to make sure if they equal to 1.

# A raw string containing JSON data
json_payload = """
{
  "event_id": "EVT_1234",
  "prediction": {
    "Yes": 0.55,
    "No": 0.45
  },
  "rationale": "A single, concise sentence summarizing the strongest evidence."
}
"""

try:
    # Parse the string into a Python dict
    data = json.loads(json_payload)

    yes_prob = data.get("prediction", {}).get("Yes", 0.0)
    no_prob = data.get("prediction", {}).get("No", 0.0)

    # 3. Stage 1 Verification: Probability Integrity Check
    # Ensure the probabilities sum up to exactly 100% (with a small margin for float precision)
    total_prob = yes_prob + no_prob
    if abs(total_prob - 1.0) > 0.001:
        return f"Probability check failed! Sums to {total_prob}, expected 1.0."
    
    return f"Probability check passed!"
except json.JSONDecodeError:
    print("Error: The provided string is not valid JSON format.")