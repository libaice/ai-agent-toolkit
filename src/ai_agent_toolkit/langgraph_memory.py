import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver  # 内存存储
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage

try:
    from .tools import search_markets, get_trending_markets, get_market_detail
except ImportError:
    from tools import search_markets, get_trending_markets, get_market_detail


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_profile: dict  # 存用户偏好


tools = [search_markets, get_trending_markets, get_market_detail]

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0).bind_tools(tools)


extractor_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)


def extract_user_preferences(messages: list, current_profile: dict) -> dict:
    """从对话中提取用户偏好，更新 profile"""

    conversation = "\n".join(
        [
            f"{m['role']}: {m['content']}"
            if isinstance(m, dict)
            else f"{m.type}: {m.content}"
            for m in messages[-6:]  # 只看最近 6 条
        ]
    )

    prompt = f"""
从以下对话中提取用户偏好信息。
只提取明确说过的信息，不要推断。

当前已知信息：
{json.dumps(current_profile, ensure_ascii=False)}

最近对话：
{conversation}

返回更新后的 JSON，格式：
{{
    "preferred_assets": ["BTC", "ETH"],
    "liquidity_preference": "低/中/高/不限",
    "capital_range": "金额或null",
    "risk_preference": "保守/激进/不限",
    "other": "其他备注"
}}

只返回 JSON，不要其他文字。
"""

    response = extractor_llm.invoke([{"role": "user", "content": prompt}])

    try:
        updated = json.loads(response.content)
        return {**current_profile, **updated}
    except Exception:
        return current_profile  # 解析失败，保持原样


def call_llm(state: AgentState):
    profile = state.get("user_profile", {})

    # 把用户偏好注入 system prompt
    profile_text = ""
    if profile:
        profile_text = f"""
用户偏好（根据历史对话记录）：
{json.dumps(profile, ensure_ascii=False, indent=2)}

请根据以上偏好过滤和推荐市场。
"""

    system = f"""
你是 Polymarket 市场分析助手。
{profile_text}
分析时关注概率合理性、成交量、流动性、截止时间。
用中文回答。
"""

    messages = [SystemMessage(content=system)] + state["messages"]

    # 只保留最近 10 条消息，防止 token 累积触发速率限制
    MAX_MESSAGES = 10
    if len(messages) > MAX_MESSAGES + 1:  # +1 是 SystemMessage
        messages = [messages[0]] + messages[-(MAX_MESSAGES):]

    response = llm.invoke(messages)
    return {"messages": [response]}



def update_memory(state: AgentState):
    """每轮对话后，更新用户画像"""
    current_profile = state.get("user_profile", {})
    updated_profile = extract_user_preferences(state["messages"], current_profile)
    return {"user_profile": updated_profile}


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "update_memory"  # 结束前先更新记忆


def build_agent_with_memory():
    graph = StateGraph(AgentState)
    
    graph.add_node("llm", call_llm)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("update_memory", update_memory)
    
    graph.set_entry_point("llm")
    
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "update_memory": "update_memory"
        }
    )
    
    graph.add_edge("tools", "llm")
    graph.add_edge("update_memory", END)
    
    # MemorySaver：跨会话保持 state
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)




def main():
    agent = build_agent_with_memory()

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
    main()