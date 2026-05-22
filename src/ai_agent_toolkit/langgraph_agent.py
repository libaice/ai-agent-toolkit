import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv(override=True)

def main() -> None:
    print("hello")


if __name__ == "__main__":
    main()