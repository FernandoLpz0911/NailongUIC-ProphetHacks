import json

# validate prediction "Yes" and "No" to make sure if they equal to 1.
# Required JSON Schema
# json_payload = """
# {
#   "event_id": "EVT_1234",
#   "prediction": {
#     "Yes": 0.55,
#     "No": 0.45
#   },
#   "rationale": "A single, concise sentence summarizing the strongest evidence."
# }
# """

def verify(json_payload):
  # Parse the string into a Python dict
  try:
    data = json.loads(json_payload)
  except:
    raise ValueError("Invalid JSON Format")
    
  if "prediction" not in data or "event_id" not in data or "rationale" not in data:
    return "Invalid JSON Format"

  yes_prob = data.get("prediction", {}).get("Yes", 0.0)
  no_prob = data.get("prediction", {}).get("No", 0.0)

  #Verification: Probability Check
  # Checking the Yes probability check
  if yes_prob < 0.0 or yes_prob > 1.0:
    return f"Probility check failed! Probability of Yes is {yes_prob}. It should be between 0.0 and 1.0"

  # Checking the No probability check
  if no_prob < 0.0 or no_prob > 1.0:
    return f"Probility check failed! Probability of No is {no_prob}. It should be between 0.0 and 1.0"
  
  # Checking the total probablity, adding both Yes and No.
  total_prob = yes_prob + no_prob
  if abs(total_prob - 1.0) > 0.001:
    return f"Probability check failed! Sums to {total_prob}, expected 1.0."
    
  return f"Probability check passed!"