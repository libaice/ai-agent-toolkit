import anthropic
import os
import json
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)


# 1/ Schema
class Decision(str, Enum):
    maker = "maker"
    taker = "taker"
    watch = "watch"
    avoid = "avoid"


class PMDecision(BaseModel):
    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str]
    risks: list[str]


# 2. input
market_facts = """
Market: BTC/USD 5-minute window
Current price: 67,450
Bid/Ask spread: 12
Order book imbalance: +0.35 (buy heavy)
Recent trades: 8 buys, 3 sells in last 60s
Volatility: 0.8% (low)
"""

# 3 call API
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="""You are a market making decision engine.
Analyze market data and output a JSON decision.
Your response must be ONLY valid JSON matching this schema:
{
  "decision": "maker" | "taker" | "watch" | "avoid",
  "confidence": float between 0.0 and 1.0,
  "evidence": list of strings,
  "risks": list of strings
}
No explanation outside the JSON.""",
    messages=[
        {"role": "user", "content": f"Analyze this market data:\n{market_facts}"}
    ]
)


raw = response.content[0].text.strip()
if raw.startswith("```json"):
    raw = raw[7:]
if raw.startswith("```"):
    raw = raw[3:]
if raw.endswith("```"):
    raw = raw[:-3]
raw = raw.strip()

print("Raw output:", raw)

decision = PMDecision.model_validate_json(raw)
print("\nParsed decision:", decision)
print("Is valid:", decision.confidence >= 0 and decision.confidence <= 1)