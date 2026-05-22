import os
try:
    from .agent import build_agent
except ImportError:
    from agent import build_agent
from dotenv import load_dotenv

load_dotenv(override=True)
agent = build_agent()


def chat(message: str):
    result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    print(result["messages"][-1].content)
    print("\n" + "─" * 50 + "\n")


if __name__ == "__main__":
    # 测试几个查询
    chat("现在 Polymarket 上最热门的市场是什么？")
    chat("帮我搜索关于美联储利率的市场，分析一下当前概率是否合理")
    chat("搜索 BTC 相关市场，哪个最值得关注")