from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage

load_dotenv(override=True)


try:
    from .tools import search_markets, get_market_detail, get_trending_markets
except ImportError:
    from tools import search_markets, get_market_detail, get_trending_markets

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


tools = [search_markets, get_market_detail, get_trending_markets]

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0
).bind_tools(tools)


SYSTEM_PROMPT = """
你是 Polymarket 市场分析助手。

你的能力：
- 搜索特定主题的预测市场
- 查看热门市场
- 获取市场详情（概率、成交量、流动性）

分析时关注：
1. 当前概率是否合理
2. 成交量和流动性是否足够
3. 截止时间
4. 是否存在明显的定价偏差

用中文回答。数据用表格或结构化格式展示。
"""

def call_llm(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] \
               + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("llm", call_llm)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("llm")

    graph.add_conditional_edges(
        "llm",
        should_continue,
        {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "llm")

    return graph.compile()
