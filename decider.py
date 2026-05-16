import csv
import hashlib
import json
import os
import time
from openai import OpenAI

# Prophet Core Imports
from ai_prophet_core import ServerAPIClient, TradeIntentRequest
from ai_prophet_core.arena import BenchmarkSession

# --- 1. LLM Client Setup ---
# Ensure OPENROUTER_API_KEY is set in your environment variables
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# --- 2. Prophet Arena API Client Setup ---
# Ensure PA_SERVER_URL and PA_SERVER_API_KEY are set
prophet_api = ServerAPIClient(
    base_url=os.environ.get("PA_SERVER_URL", "https://api.aiprophet.dev"),
    api_key=os.environ.get("PA_SERVER_API_KEY"),
    timeout=30,
)

# Bot Configuration for tracking unique experiments
CONFIG = {"strategy": "gemini-flash-2.5-dynamic", "version": "1.0"}
CONFIG_HASH = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True).encode()
).hexdigest()[:16]


def load_trading_rules(csv_path="constants.csv") -> str:
    """Reads the constants.csv and formats it into a readable string for the LLM."""
    rules_text = "Trading Constants and Execution Rules:\n"
    try:
        with open(csv_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                rules_text += f"- {row['Rule_or_Constant']}: {row['Value']} ({row['Description']})\n"
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found. Using default rules.")
    return rules_text


def get_llm_trade_decisions(market_data: list, current_portfolio: dict) -> dict:
    """
    Constructs the prompt, calls Gemini 2.5 Flash via OpenRouter, 
    and returns a dictionary containing the 'intents' array and 'reasoning'.
    """
    dynamic_rules = load_trading_rules("constants.csv")
    
    system_prompt = f"""
    You are an AI trading algorithm operating in a binary prediction market benchmark called Prophet Arena.
    Your goal is to maximize PnL by determining the best trades based on the provided market data.
    
    {dynamic_rules}
    
    OUTPUT FORMAT:
    You must return ONLY a JSON object containing an array called "intents" and a string called "reasoning". 
    Do not include markdown formatting, backticks, or explanations outside the JSON.
    
    Format:
    {{
      "reasoning": "Brief explanation of why you are making these trades.",
      "intents": [
        {{
            "market_id": "string",
            "action": "BUY" | "SELL",
            "side": "YES" | "NO",
            "shares": "string",
            "idempotency_key": ""
        }}
      ]
    }}
    """
    
    user_prompt = f"""
    CURRENT PORTFOLIO: {json.dumps(current_portfolio)}
    MARKET CANDIDATES: {json.dumps(market_data)}
    
    Based on the rules and current state, output your trade intents as JSON.
    """
    
    try:
        response = llm_client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
        
    except Exception as e:
        print(f"Error calling OpenRouter API: {e}")
        return {"intents": [], "reasoning": str(e)}


def run() -> None:
    """Main lifecycle loop for the Prophet Arena Bot."""
    print("Starting Prophet Arena LLM Bot...")
    
    with BenchmarkSession(prophet_api) as session:
        # 1. Initialize Experiment
        session.create_experiment(
            slug="gemini-flash-bot-v1",
            config_hash=CONFIG_HASH,
            config_json=CONFIG,
            n_ticks=96,
        )
        part = session.upsert_participant(
            model="custom:gemini-flash", starting_cash=10_000
        )

        # 2. Main Tick Loop
        while True:
            # Claim the next available 15-minute window
            lease = session.claim_tick()
            if not lease.available:
                if lease.reason == "experiment_completed":
                    print("Experiment completed. Exiting.")
                    break
                print(f"Tick not available. Retrying in {lease.retry_after_sec or 15}s...")
                time.sleep(lease.retry_after_sec or 15)
                continue

            print(f"\n--- Processing Tick ID: {lease.tick_id} ---")
            
            # 3. Load State (Candidates & Portfolio)
            tick = session.load_candidates(lease)
            lease = tick.lease # Update lease state
            
            # Attempt to safely serialize the portfolio
            try:
                raw_portfolio = session.get_portfolio(part.participant_idx)
                # Convert the wire model to a dict for the LLM. 
                # (Adjust .model_dump() to vars() or dict() depending on the SDK's exact Pydantic/dataclass version)
                portfolio_data = raw_portfolio.model_dump() if hasattr(raw_portfolio, 'model_dump') else vars(raw_portfolio)
            except Exception as e:
                print(f"Warning: Could not fetch portfolio: {e}")
                portfolio_data = {"error": "Could not load portfolio for this tick."}

            # Prepare market data (extracting only the necessary fields for the LLM)
            formatted_markets = []
            valid_market_ids = set()
            for m in tick.candidates.markets:
                valid_market_ids.add(m.market_id)
                formatted_markets.append({
                    "market_id": m.market_id,
                    "best_ask": float(m.quote.best_ask) if m.quote.best_ask else None,
                    "best_bid": float(m.quote.best_bid) if m.quote.best_bid else None,
                })

            # 4. Get LLM Decision
            llm_response = get_llm_trade_decisions(formatted_markets, portfolio_data)
            raw_intents = llm_response.get("intents", [])
            reasoning = llm_response.get("reasoning", "No reasoning provided.")
            
            print(f"LLM Reasoning: {reasoning}")
            print(f"Generated {len(raw_intents)} intents.")

            # 5. Map LLM Output to SDK Wire Models
            intents = []
            for intent_data in raw_intents:
                # Sanity check: Ensure the LLM didn't hallucinate a market ID
                if intent_data.get("market_id") not in valid_market_ids:
                    print(f"Skipping invalid market_id: {intent_data.get('market_id')}")
                    continue
                    
                intents.append(TradeIntentRequest(
                    market_id=intent_data["market_id"],
                    action=intent_data["action"],
                    side=intent_data["side"],
                    shares=str(intent_data["shares"]), # Force string-encoded decimal
                    idempotency_key="",                # SDK auto-fills this
                ))

            # 6. Execute and Finalize
            # Storing the reasoning in the plan JSON so it shows up in the Live Dashboard!
            session.put_plan(lease, part.participant_idx, {"llm_reasoning": reasoning})
            
            if intents:
                session.submit_intents(lease, part.participant_idx, intents)
                
            session.finalize(lease, part.participant_idx)
            session.complete_tick(lease)
            
            print("Tick completed successfully.")


if __name__ == "__main__":
    # Ensure mandatory environment variables exist before starting the infinite loop
    missing_vars = [var for var in ["OPENROUTER_API_KEY", "PA_SERVER_API_KEY"] if not os.environ.get(var)]
    if missing_vars:
        print(f"ERROR: Missing environment variables: {', '.join(missing_vars)}")
        exit(1)
        
    run()