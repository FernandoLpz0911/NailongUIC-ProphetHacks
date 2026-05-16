import csv
import json

def load_trading_rules(csv_path="constants.csv") -> str:
    """
    Reads the constants.csv and formats it into a readable string for the LLM prompt.
    """
    rules_text = "Trading Constants and Execution Rules:\n"
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            rules_text += f"- {row['Rule_or_Constant']}: {row['Value']} ({row['Description']})\n"
    return rules_text

def get_llm_trade_decisions(market_data: dict, current_portfolio: dict) -> list[dict]:
    """
    Constructs the prompt, calls the LLM, and returns a list of TradeIntentRequests.
    """
    dynamic_rules = load_trading_rules("constants.csv")
    
    system_prompt = f"""
    You are an AI trading algorithm operating in a binary prediction market benchmark called Prophet Arena.
    Your goal is to maximize PnL by determining the best trades based on the provided market data.
    
    {dynamic_rules}
    
    OUTPUT FORMAT:
    You must return ONLY a JSON array of trade intents. Do not include markdown formatting or explanations.
    Each object in the array must strictly follow this structure:
    {{
        "market_id": "string", // Extracted from the provided market candidates
        "action": "BUY" | "SELL",
        "side": "YES" | "NO",
        "shares": "string", // Must be a string-encoded decimal (e.g., "10")
        "idempotency_key": "" // Leave empty, the SDK auto-generates this
    }}
    """
    
    # 3. Construct the User Prompt (Market state and Portfolio)
    user_prompt = f"""
    CURRENT PORTFOLIO: {json.dumps(current_portfolio)}
    MARKET CANDIDATES: {json.dumps(market_data)}
    
    Based on the rules and current state, output your trade intents as a JSON array.
    """
    
    # 4. Mock LLM Call (replace with actual OpenAI/Anthropic/Gemini SDK call)
    # response = llm_client.chat.completions.create(
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt}
    #     ],
    #     response_format={"type": "json_object"}
    # )
    
    # 5. Parse and return
    # intents = json.loads(response.choices[0].message.content)
    # return intents
    pass