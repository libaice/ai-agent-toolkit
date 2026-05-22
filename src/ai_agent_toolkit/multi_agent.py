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


# 2. define sub agent

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

