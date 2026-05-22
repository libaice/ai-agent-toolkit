import os
import time
try:
    from .agent import build_agent
except ImportError:
    from agent import build_agent
from dotenv import load_dotenv

try:
    from .langgraph_memory import build_agent_with_memory
except ImportError:
    from langgraph_memory import build_agent_with_memory


load_dotenv(override=True)
# agent = build_agent()

agent = build_agent_with_memory()


# def chat(message: str):
#     result = agent.invoke({"messages": [{"role": "user", "content": message}]})
#     print(result["messages"][-1].content)
#     print("\n" + "─" * 50 + "\n")


# if __name__ == "__main__":
#     # 测试几个查询
#     chat("现在 Polymarket 上最热门的市场是什么？")
#     chat("帮我搜索关于美联储利率的市场，分析一下当前概率是否合理")
#     chat("搜索 BTC 相关市场，哪个最值得关注")



def chat(message: str, user_id: str):
    # thread_id 决定哪个用户的 memory
    config = {"configurable": {"thread_id": user_id}}
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config
    )
    
    # 打印回答
    print(f"\nAgent: {result['messages'][-1].content}")
    
    # 打印当前记住的用户偏好
    profile = result.get("user_profile", {})
    if profile:
        import json
        print(f"\n[Memory] {json.dumps(profile, ensure_ascii=False)}")


if __name__ == "__main__":
    user = "bruce_001"
    
    # 第一轮：告诉 agent 偏好
    print("[1/4] 第一轮对话...")
    chat("我主要关注 BTC 和 ETH 相关的市场，偏好低流动性", user)
    
    print("[等待 35s 避免速率限制...]")
    time.sleep(35)
    
    # 第二轮：agent 应该记住
    print("[2/4] 第二轮对话...")
    chat("帮我找几个值得关注的市场", user)
    
    print("[等待 35s 避免速率限制...]")
    time.sleep(35)
    
    # 第三轮：新信息
    print("[3/4] 第三轮对话...")
    chat("我的资金大概 3000 USDC，风格偏保守", user)
    
    print("[等待 35s 避免速率限制...]")
    time.sleep(35)
    
    # 第四轮：综合所有记忆
    print("[4/4] 第四轮对话...")
    chat("今天有什么机会？", user)
