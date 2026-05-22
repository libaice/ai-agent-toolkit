# evaluator.py

from langchain_anthropic import ChatAnthropic
import json
from dotenv import load_dotenv

load_dotenv(override=True)  # 必须在 LLM 初始化之前

try:
    from .agent_with_rag import agent
except ImportError:
    from agent_with_rag import agent

judge_llm = ChatAnthropic(model="claude-sonnet-4-20250514")



def evaluate_response(
    question: str, agent_answer: str, reference_answer: str = None
) -> dict:
    """
    让 LLM 给 agent 的回答打分
    """

    if reference_answer:
        prompt = f"""
评估以下 AI 助手的回答质量。

问题：{question}

标准答案：{reference_answer}

AI 回答：{agent_answer}

请从以下维度打分（1-5分）：
1. 准确性：与标准答案的吻合程度
2. 完整性：是否覆盖了关键信息
3. 幻觉：是否编造了不存在的信息（5=无幻觉，1=严重幻觉）

只返回 JSON：
{{
    "accuracy": 分数,
    "completeness": 分数,
    "hallucination": 分数,
    "reasoning": "简短说明"
}}
"""
    else:
        # 没有标准答案，只评估质量
        prompt = f"""
评估以下 AI 助手的回答质量。

问题：{question}

AI 回答：{agent_answer}

请从以下维度打分（1-5分）：
1. 相关性：回答是否切题
2. 清晰度：表达是否清晰
3. 幻觉：是否编造信息（5=无幻觉，1=严重幻觉）

只返回 JSON：
{{
    "relevance": 分数,
    "clarity": 分数,
    "hallucination": 分数,
    "reasoning": "简短说明"
}}
"""

    response = judge_llm.invoke([{"role": "user", "content": prompt}])

    try:
        return json.loads(response.content)
    except:
        return {"error": "解析失败", "raw": response.content}


def evaluate_tool_usage(
    question: str, expected_tools: list[str], actual_messages: list
) -> dict:
    """
    检查 agent 是否调用了预期的工具
    """
    # 从消息历史里提取实际调用的工具
    actual_tools = []
    for msg in actual_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                actual_tools.append(tc["name"])

    # 计算覆盖率
    expected_set = set(expected_tools)
    actual_set = set(actual_tools)

    hit = expected_set & actual_set  # 该用的用了
    miss = expected_set - actual_set  # 该用的没用
    extra = actual_set - expected_set  # 多用的

    score = len(hit) / len(expected_set) if expected_set else 1.0

    return {
        "score": round(score, 2),
        "hit": list(hit),
        "missed": list(miss),
        "extra": list(extra),
        "passed": score == 1.0,
    }


TEST_CASES = [
    {
        "id": "tc_001",
        "question": "GhostGuard 的月费是多少",
        "expected_tools": ["search_knowledge_base"],
        "reference_answer": "GhostGuard 月费 $299，定制集成 $2500，咨询 $150/小时",
    },
    {
        "id": "tc_002",
        "question": "现在 BTC 相关市场有哪些",
        "expected_tools": ["search_markets"],
        "reference_answer": None,  # 实时数据，没有固定答案
    },
    {
        "id": "tc_003",
        "question": "V2 的签名风险怎么防",
        "expected_tools": ["search_knowledge_base"],
        "reference_answer": "使用 EIP-1271 钱包，配合 GhostGuard 监控签名状态",
    },
    {
        "id": "tc_004",
        "question": "查一下热门市场，结合做市商风险给建议",
        # 这个问题需要同时用两个工具
        "expected_tools": ["get_trending_markets", "search_knowledge_base"],
        "reference_answer": None,
    },
]


def run_eval():
    results = []

    for tc in TEST_CASES:
        print(f"运行 {tc['id']}: {tc['question'][:30]}...")

        # 运行 agent
        output = agent.invoke({
            "messages": [{"role": "user", "content": tc["question"]}]
        })

        answer   = output["messages"][-1].content
        messages = output["messages"]

        # 工具评估
        tool_eval = evaluate_tool_usage(
            tc["question"],
            tc["expected_tools"],
            messages
        )

        # 答案评估
        answer_eval = evaluate_response(
            tc["question"],
            answer,
            tc.get("reference_answer")
        )

        result = {
            "id":          tc["id"],
            "question":    tc["question"],
            "tool_eval":   tool_eval,
            "answer_eval": answer_eval,
            "passed":      tool_eval["passed"]
        }
        results.append(result)

        # 打印单条结果
        status = "✅" if tool_eval["passed"] else "❌"
        print(f"  {status} 工具分: {tool_eval['score']}")
        if not tool_eval["passed"]:
            print(f"     缺失工具: {tool_eval['missed']}")

    # 汇总
    total   = len(results)
    passed  = sum(1 for r in results if r["passed"])
    rate    = passed / total * 100

    print(f"\n{'─'*40}")
    print(f"通过率: {passed}/{total} ({rate:.0f}%)")

    # 保存结果
    with open("eval_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    run_eval()