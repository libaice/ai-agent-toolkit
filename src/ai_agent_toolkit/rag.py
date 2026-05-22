from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
import os


DOCUMENTS = [
    """
    GhostGuard 使用指南
    
    GhostGuard 是 EOA maker 签名生命周期追踪工具。
    主要风险：Polymarket V2 中，签名订单没有取消机制，
    100 个区块暂停期约 3.5 分钟。
    这段时间内，旧签名仍然有效，可能被恶意执行。
    
    检测方法：监控链上事件，对比签名时间戳和当前区块高度。
    如果差值超过阈值，发出警报。
    
    定价：$299/月，$2500 定制集成，$150/小时咨询。
    """,
    
    """
    Polymarket V2 技术要点
    
    V2 相比 V1 的核心变化：
    1. fillOrder 方法被移除
    2. 订单通过 Builder Relayer 路由
    3. EOA maker 需要 EIP-712 签名
    4. 撤单需要通过 cancel registry
    
    幽灵成交（Ghost Fill）：V1 的问题，V2 已修复。
    新风险：过期签名（Stale Signed Orders）。
    
    Builder Relayer：负责撮合和路由，
    Rust SDK：rs-builder-relayer-client，
    目前 Polymarket 生态中唯一的 Rust V2 SDK。
    """,
    
    """
    Polymarket 做市商风险管理
    
    主要风险类型：
    1. 签名风险：签名过期或被重放
    2. 流动性风险：单边成交导致持仓集中
    3. 预言机风险：结算价格操纵
    4. 网络风险：Polygon 网络拥堵导致撤单失败
    
    缓解方案：
    - 使用 EIP-1271 钱包代替 EOA
    - 实时监控签名状态（GhostGuard）
    - 设置最大单边持仓限额
    - 保持 USDC 储备应对紧急撤单 gas 费
    """
]



# Chuck 
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 每块最多 500 字
    chunk_overlap=50,     # 块之间重叠 50 字，避免切断语义
)

chunks = splitter.create_documents(DOCUMENTS)
print(f"切成 {len(chunks)} 块")



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # 支持中文，轻量，够用
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./knowledge_base"  # 存到本地
)


print("知识库建完")

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # 每次返回最相关的 3 块
)


@tool
def search_knowledge_base(query: str) -> str:
    """
    搜索内部知识库。
    包含 GhostGuard 文档、Polymarket V2 技术细节、做市商风险管理指南。
    当用户询问这些相关内容时调用。
    """
    docs = retriever.invoke(query)
    
    if not docs:
        return "知识库中没有找到相关信息"
    
    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"[片段 {i}]\n{doc.page_content}")
    
    return "\n\n".join(results)