import httpx
from langchain_core.tools import tool

GAMMA_BASE = "https://gamma-api.polymarket.com"


@tool
def search_markets(query: str, limit: int = 5) -> str:
    """
    搜索 Polymarket 市场。
    query: 关键词，比如 "bitcoin" "election" "fed"
    返回市场列表，包含名称、ID、当前概率、成交量。
    """
    resp = httpx.get(
        f"{GAMMA_BASE}/markets", params={"q": query, "limit": limit, "active": True}
    )
    markets = resp.json()

    if not markets:
        return "没有找到相关市场"

    result = []
    for m in markets:
        result.append(
            f"- {m['question']} (ID: {m.get('id', 'N/A')})\n"
            f"  概率: {m.get('outcomePrices', 'N/A')}\n"
            f"  成交量: ${float(m.get('volume', 0)):,.0f}\n"
            f"  截止: {m.get('endDate', 'N/A')}"
        )
    return "\n".join(result)


@tool
def get_market_detail(market_id: str) -> str:
    """
    获取某个市场的详细信息。
    market_id: 市场 ID，从 search_markets 结果里取。
    """
    resp = httpx.get(f"{GAMMA_BASE}/markets/{market_id}")
    if resp.status_code != 200:
        return f"获取市场详情失败，HTTP 状态码: {resp.status_code}"
    
    m = resp.json()
    if "question" not in m:
        return f"未找到 ID 为 {market_id} 的市场，请确保使用的是正确的市场 ID（通常为一串数字）。"

    return (
        f"市场: {m['question']}\n"
        f"ID: {m.get('id', 'N/A')}\n"
        f"描述: {m.get('description', '')[:200]}\n"
        f"概率: {m.get('outcomePrices')}\n"
        f"成交量: ${float(m.get('volume', 0)):,.0f}\n"
        f"流动性: ${float(m.get('liquidity', 0)):,.0f}\n"
        f"截止时间: {m.get('endDate')}"
    )


@tool
def get_trending_markets(limit: int = 5) -> str:
    """
    获取当前成交量最高的热门市场。
    """
    resp = httpx.get(
        f"{GAMMA_BASE}/markets",
        params={"limit": limit, "active": True, "order": "volume", "ascending": False},
    )
    markets = resp.json()

    result = []
    for m in markets:
        result.append(
            f"- {m['question']} (ID: {m.get('id', 'N/A')})\n"
            f"  概率: {m.get('outcomePrices')}\n"
            f"  成交量: ${float(m.get('volume', 0)):,.0f}"
        )
    return "\n".join(result)
