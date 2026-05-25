import anthropic
import os
import json
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

tools = [
    
    {
        "name": "get_spot_price",
        "description": "Get current spot price for a symbol",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "e.g. BTC, ETH, AAPL"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_orderbook",
        "description": "Get current orderbook for a Polymarket market",
        "input_schema": {
            "type": "object",
            "properties": {
                "market_id": {
                    "type": "string",
                    "description": "Polymarket market ID"
                }
            },
            "required": ["market_id"]
        }
    },
    {
        "name": "get_recent_trades",
        "description": "Get recent trades for a market",
        "input_schema": {
            "type": "object",
            "properties": {
                "market_id": {
                    "type": "string"
                },
                "limit": {
                    "type": "integer",
                    "default": 10
                }
            },
            "required": ["market_id"]
        }
    }
] 



def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "get_spot_price":
        prices = {"BTC": 67450, "ETH": 3200, "AAPL": 189.5}
        symbol = tool_input["symbol"]
        price = prices.get(symbol, "unknown")
        return json.dumps({"symbol": symbol, "price": price, "unit": "USD"})

    elif tool_name == "get_orderbook":
        return json.dumps({
            "market_id": tool_input["market_id"],
            "bids": [{"price": 0.6 - i*0.01, "size": 100} for i in range(20)],
            "asks": [{"price": 0.61 + i*0.01, "size": 80} for i in range(20)]
        })

    elif tool_name == "get_recent_trades":
        raise Exception("API rate limit exceeded")


def run_agent(user_message: str):
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        print(f"\nStop reason: {response.stop_reason}")
        
        # 模型决定不调工具，直接回答
        if response.stop_reason == "end_turn":
            final_text = next(
                (b.text for b in response.content if hasattr(b, 'text')),
                "No response"
            )
            print(f"Final answer: {final_text}")
            return final_text
        
        # 模型想调工具
        if response.stop_reason == "tool_use":
            # 把模型的回应加入对话历史
            messages.append({"role": "assistant", "content": response.content})
            
            # 处理每个工具调用
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\nModel wants to call: {block.name}")
                    print(f"With params: {block.input}")
                    
                    # 你的代码执行工具
                    try:
                        result = execute_tool(block.name, block.input)
                        print(f"Tool result: {result[:100]}...")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                    except Exception as e:
                        # 工具失败时怎么处理
                        print(f"Tool failed: {e}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error: {str(e)}",
                            "is_error": True
                        })
            
            # 把工具结果返回给模型
            messages.append({"role": "user", "content": tool_results})

# 4. 测试 4 种场景
print("=" * 50)
print("Test 1: 正常价格查询")
run_agent("BTC 现在多少钱？")

print("\n" + "=" * 50)
print("Test 2: 工具选择测试")
run_agent("帮我分析 market_id_12345 这个市场的订单簿")

print("\n" + "=" * 50)
print("Test 3: 工具失败场景")
run_agent("给我看 market_id_999 的最近交易记录")

print("\n" + "=" * 50)
print("Test 4: 不存在的 symbol（观察 hallucination）")
run_agent("FAKETOKEN 的价格是多少？")