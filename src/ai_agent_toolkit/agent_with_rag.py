from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
load_dotenv(override=True)


try:
    from .tools import search_markets, get_trending_markets
    from .rag import search_knowledge_base
except ImportError:
    from tools import search_markets, get_trending_markets
    from rag import search_knowledge_base



tools = [
    search_markets,
    get_trending_markets,
    search_knowledge_base,  # ← 加进来
]

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0).bind_tools(tools)

SYSTEM = """
你是 Polymarket 专家助手。

你有两类能力：
1. 实时数据：搜索当前市场、价格、成交量
2. 内部知识：GhostGuard 文档、V2 技术细节、风险管理

遇到技术问题先查知识库，遇到市场数据再查实时 API。


"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def call_llm(state: AgentState):
    messages = [SystemMessage(content=SYSTEM)] + state["messages"]
    return {"messages": [llm.invoke(messages)]}


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")

agent = graph.compile()



if __name__ == "__main__":
    def chat(query: str):
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
        print(response["messages"][-1].content)

    # 记忆：用户说过的 V2 知识、安全意识、保守风格
    chat("Polymarket V2 有什么风险？")
    chat("我比较保守，能推荐几个安全的市场吗？")

