import anthropic
import os
import json
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

# 定义三个 specialist agent 的 system prompt
MARKET_AGENT_PROMPT = """You are MarketAgent, specialized in analyzing order books 
and market microstructure. You analyze bid/ask spreads, order imbalances, and 
liquidity. Respond with structured market analysis."""

NEWS_AGENT_PROMPT = """You are NewsAgent, specialized in finding and analyzing 
news that could affect prediction market outcomes. You identify key events, 
sentiment, and probability shifts."""

RISK_AGENT_PROMPT = """You are RiskAgent, the final decision maker. You receive 
analysis from MarketAgent and NewsAgent, then output a final decision:
maker/taker/watch/avoid with confidence score and reasoning."""

def run_specialist(system_prompt: str, task: str) -> str:
    """运行一个专门的 agent"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": task}]
    )
    return response.content[0].text

def run_triage_agent(user_query: str) -> str:
    """主 Agent：判断需要哪些 specialists，协调结果"""
    
    print("\n[Triage] Analyzing query...")
    
    # 主 agent 判断需要哪个 specialist
    triage_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system="""You are a triage agent. Given a user query about prediction markets,
decide which specialists to consult. Reply with JSON:
{"specialists": ["market", "news", "risk"], "reason": "..."}
Only include needed specialists. Always include "risk" as final step.""",
        messages=[{"role": "user", "content": user_query}]
    )
    
    raw = triage_response.content[0].text.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    
    triage = json.loads(raw)
    print(f"[Triage] Will consult: {triage['specialists']}")
    
    results = {}
    
    # 按顺序调用需要的 specialists
    if "market" in triage["specialists"]:
        print("\n[MarketAgent] Analyzing...")
        results["market"] = run_specialist(
            MARKET_AGENT_PROMPT,
            f"Analyze market conditions for: {user_query}"
        )
        print(f"[MarketAgent] Done: {results['market'][:100]}...")
    
    if "news" in triage["specialists"]:
        print("\n[NewsAgent] Checking news...")
        results["news"] = run_specialist(
            NEWS_AGENT_PROMPT,
            f"Find relevant news for: {user_query}"
        )
        print(f"[NewsAgent] Done: {results['news'][:100]}...")
    
    # RiskAgent 总是最后，汇总所有结果
    if "risk" in triage["specialists"]:
        print("\n[RiskAgent] Making final decision...")
        context = "\n".join([f"{k} analysis: {v}" for k, v in results.items()])
        final = run_specialist(
            RISK_AGENT_PROMPT,
            f"User query: {user_query}\n\n{context}\n\nMake final decision."
        )
        print(f"\n[Final Decision]: {final}")
        return final
    
    return "No risk assessment made"

# 测试
print("=" * 50)
result = run_triage_agent(
    "Should I make markets on 'Will BTC be above 70k by end of month?'"
)
