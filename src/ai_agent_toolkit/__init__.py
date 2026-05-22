import os
import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def main() -> None:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system="你是一个交易助手",
        messages=[
            {"role": "user", "content": "BTC 最近走势怎么样"}
        ]
    )

    print(response)
