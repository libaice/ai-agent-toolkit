import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

load_dotenv(override=True)


# 1 Agent State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def get_price(symbol: str) -> str:
    """获取资产当前价格。symbol 是资产符号如 BTC、ETH。"""
    prices = {"BTC": "67432", "ETH": "3521"}
    return prices.get(symbol, "未找到")


@tool
def get_24h_change(symbol: str) -> str:
    """获取资产过去 24 小时涨跌幅。"""
    changes = {"BTC": "+3.2%", "ETH": "-1.1%"}
    return changes.get(symbol, "未找到")


# 2. Agent tool
tools = [get_price, get_24h_change]

# 3. LLM init with tools
llm = ChatAnthropic(model="claude-sonnet-4-20250514").bind_tools(tools)


# 4. build state nodes
def call_llm(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    """路由函数：决定下一步去哪个节点"""
    last = state["messages"][-1]
    if last.tool_calls:
        return "execute_tools"
    return END


# 5 .build graph
graph = StateGraph(AgentState)

graph.add_node("call_llm", call_llm)
graph.add_node("execute_tools", ToolNode(tools))


graph.set_entry_point("call_llm")

graph.add_conditional_edges(
    "call_llm",  # 从这个节点出发
    should_continue,  # 用这个函数决定去哪
    {
        "execute_tools": "execute_tools",  # 返回值 → 目标节点
        END: END,
    },
)

graph.add_edge("execute_tools", "call_llm")


def main() -> None:
    app = graph.compile()
    # result = app.invoke(
        # {"messages": [{"role": "user", "content": "BTC 和 ETH 今天哪个更强"}]}
    # )
    # print(result["messages"][-1].content)

    for chunk in app.stream({"messages": [{"role": "user", "content": "分析 BTC 和 ETH 今天哪个更强"}]}):
        for node_name, node_output in chunk.items():
            print(f"节点 [{node_name}] 输出:", node_output)


if __name__ == "__main__":
    main()


