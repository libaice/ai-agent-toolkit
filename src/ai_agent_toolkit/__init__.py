import os
import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

messages = [{"role": "user", "content": "BTC 现在多少钱"}]


tools = [
    {
        "name": "get_price",
        "description": "获取某个资产的当前价格。当用户问价格时使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "资产符号，比如 BTC、ETH"}
            },
            "required": ["symbol"],
        },
    }
]


def execute_tool(name: str, args: dict) -> str:
    if name == "get_price":
        prices = {"BTC": "67432", "ETH": "3521"}
        return prices.get(args["symbol"], "未找到")

    if name == "get_24h_change":
        changes = {"BTC": "+3.2%", "ETH": "-1.1%"}
        return changes.get(args["symbol"], "未找到")

    return "工具不存在"


def run_agent(user_input: str):
    messages = [
        {"role": "user", "content": user_input}
    ]
    
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            tools=tools,
            messages=messages
        )
        
        if response.stop_reason == "end_turn":
            return response.content[0].text
        
        if response.stop_reason == "tool_use":
            messages.append({
                "role": "assistant",
                "content": response.content
            })
            
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            messages.append({
                "role": "user",
                "content": tool_results
            })
            


def get_price(symbol: str) -> str:
    prices = {"BTC": "77424", "ETH": "2128"}
    return prices.get(symbol, "Not Found")


def main() -> None:

    answer = run_agent("帮我分析一下 BTC 和 ETH，哪个今天更值得关注")
    print(answer)

    # response = client.messages.create(
    #     model="claude-sonnet-4-20250514",
    #     max_tokens=1000,
    #     tools=tools,
    #     messages=messages,
    # )

    # if response.stop_reason == "tool_use":
    #     tool_call = response.content[0]
    #     tool_name = tool_call.name  # "get_price"
    #     tool_args = tool_call.input  # {"symbol": "BTC"}
    #     tool_id = tool_call.id  # "call_abc123"

    #     # execute
    #     result = get_price(**tool_args)
    #     print("get the price from cex", result)

    #     messages.append({"role": "assistant", "content": response.content})
    # messages.append(
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "tool_result", "tool_use_id": tool_id, "content": result}
    #         ],
    #     }
    # )
    # final = client.messages.create(
    #     model="claude-sonnet-4-20250514",
    #     max_tokens=1000,
    #     tools=tools,
    #     messages=messages,
    # )

    # print(final.content[0].text)
