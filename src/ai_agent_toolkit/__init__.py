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


def get_price(symbol: str) -> str:
    prices = {"BTC": "77424", "ETH": "2128"}
    return prices.get(symbol, "Not Found")


def main() -> None:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        tools=tools,
        messages=messages,
    )

    if response.stop_reason == "tool_use":
        tool_call = response.content[0]
        tool_name = tool_call.name  # "get_price"
        tool_args = tool_call.input  # {"symbol": "BTC"}
        tool_id = tool_call.id  # "call_abc123"

        # execute
        result = get_price(**tool_args)
        print("get the price from cex", result)

        messages.append({"role": "assistant", "content": response.content})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_id, "content": result}
            ],
        }
    )
    final = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        tools=tools,
        messages=messages,
    )

    print(final.content[0].text)
