import anthropic
import json
import os
from pydantic import BaseModel, Field
from enum import Enum

from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)


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


def input_guardrail(market_data: dict) -> tuple[bool, str]:
    required_fields = ["price", "spread", "volume"]
    missing = [f for f in required_fields if f not in market_data]
    if missing:
        return False, f"Missing required fields: {missing}"
    if market_data.get("spread", 0) > 100:
        return False, "Spread too wide, market conditions unreliable"
    return True, "OK"


def output_guardrail(decision: PMDecision) -> tuple[bool, str]:
    # 规则 1：低置信度强制降级为 watch/avoid
    if decision.confidence < 0.6:
        return False, f"Confidence {decision.confidence} too low for active decision"
    # 规则 2：must have at least 2 evidence points
    if len(decision.evidence) < 2:
        return False, "Insufficient evidence provided"
    return True, "OK"


def run_with_guardrails(market_data: dict, max_retries: int = 2):
    valid, reason = input_guardrail(market_data)
    if not valid:
        print(f"Input guardrail failed: {reason}")
        return None

    print("[Input guardrail passed]")

    # init messages
    messages = [
        {"role": "user", "content": f"Analyze: {json.dumps(market_data)}"}
    ]
    for attempt in range(max_retries + 1):
        print(f"\nAttempt {attempt + 1}...")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system="""Output ONLY valid JSON  for market decision No ```json. No explanation. :
                Just the JSON object, nothing else.
                {
                "decision": "maker" | "taker" | "watch" | "avoid",
                "confidence": 0.0-1.0,
                "evidence": ["at least", "two points"],
                "risks": ["list of risks"]
                }""",
            messages=[
                {"role": "user", "content": f"Analyze: {json.dumps(market_data)}"}
            ],
        )
        raw = response.content[0].text.strip()

      

        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        print("Model response:", raw)

        try:
            decision = PMDecision.model_validate_json(raw)
        except Exception as e:
            print(f"Parse error: {e}")
            # 把失败原因加进 messages，下次重试带着
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Your output was invalid JSON: {e}. Output ONLY raw JSON, no markdown."})
            continue
        
        valid, reason = output_guardrail(decision)
        if valid:
            print(f"[Output guardrail passed]")
            print(f"Final decision: {decision}")
            return decision
        else:
            print(f"[OUTPUT GUARDRAIL FAILED]: {reason}")
            if attempt < max_retries:
                print("Retrying with stricter instruction...")
        print("Max retries exceeded, returning AVOID")
        return PMDecision(
            decision=Decision.avoid,
            confidence=0.0,
            evidence=[],
            risks=["Max retries exceeded"]
        )


print("=" * 50)
print("Test 1: 正常数据")

# run_with_guardrails({"price": 67450, "spread": 15, "volume": 1200000, "imbalance": 0.3})

print("\n" + "=" * 50)
print("Test 2: 缺少字段（input guardrail 触发）")
# run_with_guardrails({
#     "price": 67450
#     # 缺少 spread 和 volume
# })


print("\n" + "=" * 50)
print("Test 3: 故意让模型输出低置信度（output guardrail 触发）")
run_with_guardrails({
    "price": 67450,
    "spread": 20,  # 高 spread 暗示不确定
    "volume": 150,  # 低 volume
    "imbalance": 0.01
})
