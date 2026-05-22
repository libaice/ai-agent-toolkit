import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

load_dotenv(override=True)


# 1. subagent
@tool
def search_market_data(symbol: str) -> str:
    """搜索市场数据和新闻"""
    return f"{symbol} 近期成交量上升 30%，机构买入信号明显"


@tool
def write_report(data: str) -> str:
    """根据数据生成分析报告"""
    return f"分析报告：\n{data}\n结论：短期看涨"


@tool
def send_email(content: str, to: str) -> str:
    """发送邮件"""
    return f"邮件已发送至 {to}"


# 2. define State
class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str  # supervisor 决定下一步派给谁
    final_report: str  # 最终结果


# 3. create sub agent
llm = ChatAnthropic(model="claude-sonnet-4-20250514")


def make_agent(tools: list, system_prompt: str):
    """工厂函数：生产一个绑定了工具和 prompt 的 agent，并在内部自动运行工具"""
    agent_llm = llm.bind_tools(tools)

    def agent_node(state: MultiAgentState):
        from langchain_core.messages import SystemMessage, ToolMessage

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        
        while True:
            response = agent_llm.invoke(messages)
            if not response.tool_calls:
                break
            
            messages.append(response)
            for tool_call in response.tool_calls:
                # Find the matching tool from our tools list
                tool_obj = next(t for t in tools if t.name == tool_call["name"])
                tool_output = tool_obj.invoke(tool_call["args"])
                messages.append(
                    ToolMessage(
                        content=str(tool_output),
                        tool_call_id=tool_call["id"]
                    )
                )
        
        return {"messages": [response]}

    return agent_node


research_agent = make_agent(
    tools=[search_market_data],
    system_prompt="你是市场研究员。只需调用 search_market_data 搜集指定资产的数据，并将搜集到的原始数据/事实提供给分析师，不要自己编写最终报告或解释其他任务（例如发送邮件）。",
)

writing_agent = make_agent(
    tools=[write_report],
    system_prompt="你是分析师。负责将研究员搜集到的原始数据整理并调用 write_report 生成分析报告。只需输出生成的报告内容，不要做其他任务。",
)

sending_agent = make_agent(
    tools=[send_email],
    system_prompt="你是助手。负责调用 send_email 把分析师编写好的报告发送到指定的邮箱地址。只需输出发送状态或结果。",
)

# ── 4. Supervisor ─────────────────────────────

MEMBERS = ["research", "writing", "sending"]

supervisor_prompt = f"""
你是任务协调员。你的职责是指导 specialized agents 协作完成用户的任务。
成员角色：
- research: 搜集市场原始数据和新闻。
- writing: 整理研究数据并撰写分析报告。
- sending: 将最终报告发送邮件。

工作流规则：
1. 首先派给 `research` 搜集数据。
2. 得到原始数据后，派给 `writing` 撰写报告。
3. 撰写好报告后，派给 `sending` 发送邮件。
4. 邮件发送完成后，返回 `FINISH` 结束任务。

只返回可选成员中的一个词：{MEMBERS + ["FINISH"]}。
不要输出任何其他多余文本。
"""


def supervisor_node(state: MultiAgentState):
    supervisor_llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    from langchain_core.messages import SystemMessage

    response = supervisor_llm.invoke(
        [SystemMessage(content=supervisor_prompt)] + state["messages"]
    )

    content = response.content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            elif hasattr(part, "text"):
                text_parts.append(part.text)
            else:
                text_parts.append(str(part))
        next_agent = "".join(text_parts).strip()
    else:
        next_agent = str(content).strip()

    # 确保 next_agent 是有效的可选成员或 FINISH
    valid_options = MEMBERS + ["FINISH"]
    if next_agent not in valid_options:
        for option in valid_options:
            if option.lower() in next_agent.lower():
                next_agent = option
                break

    return {"next": next_agent}


# 5. build graph
graph = StateGraph(MultiAgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("research", research_agent)
graph.add_node("writing", writing_agent)
graph.add_node("sending", sending_agent)


graph.set_entry_point("supervisor")


graph.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {"research": "research", "writing": "writing", "sending": "sending", "FINISH": END},
)

for member in MEMBERS:
    graph.add_edge(member, "supervisor")


def main() -> None:
    app = graph.compile()
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": "研究一下 BTC 市场，写份报告，发到 bruce@example.com",
            }
        ],
        "next": "",
        "final_report": "",
    }
    for chunk in app.stream(inputs):
        for node_name, node_output in chunk.items():
            print(f"\n--- 节点 [{node_name}] ---")
            if "next" in node_output:
                print(f"Supervisor 决策: 下一步派给 -> {node_output['next']}")
            if "messages" in node_output:
                print(f"最新消息: {node_output['messages'][-1].content}")


if __name__ == "__main__":
    main()
