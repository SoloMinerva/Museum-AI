"""
查询路由器：用轻量 LLM 对用户问题做意图分类，决定检索策略。

分类结果：
  - chat:    闲聊/通用问题，LLM 直接回答，不查知识库
  - simple:  简单文物查询（精确查找、单一属性），走 LightRAG naive 模式
  - complex: 复杂问题（跨文物对比、关系推理、综合分析），走 LightRAG mix 模式
"""

from __future__ import annotations

from src.models.chat import select_model
from src.utils import logger

ROUTER_PROMPT = """你是一个问题分类器。根据用户的问题，判断它属于以下哪个类别，只返回类别名称，不要返回其他内容。

类别：
- chat: 闲聊、问候、感谢、与博物馆/文物完全无关的问题
- simple: 查询具体某个文物的信息、单一属性（名称、年代、尺寸、出土地点等）
- complex: 需要对比多个文物、跨朝代分析、关系推理、综合归纳的问题

示例：
"你好" → chat
"今天天气怎么样" → chat
"谢谢你的回答" → chat
"曾侯乙编钟是什么" → simple
"越王勾践剑的长度是多少" → simple
"新石器时代有哪些陶器" → simple
"湖北省博物馆有哪些青铜器" → simple
"从商代到战国青铜器风格怎么演变的" → complex
"两个博物馆各有什么镇馆之宝" → complex
"曾侯乙墓出土的文物反映了怎样的音乐文化" → complex
"古代玉器在不同朝代的用途有什么变化" → complex

用户问题：{question}

类别："""

# 类别 → LightRAG mode 映射
ROUTE_TO_MODE = {
    "chat": None,       # 不查库
    "simple": "naive",  # 向量检索
    "complex": "mix",   # 图谱+向量
}

VALID_CATEGORIES = set(ROUTE_TO_MODE.keys())


def classify_query(question: str, model_spec: str | None = None) -> str:
    """
    用 LLM 对问题进行分类。

    Args:
        question: 用户问题
        model_spec: 模型规格，默认使用 fast_model

    Returns:
        分类结果: "chat" / "simple" / "complex"
    """
    if not model_spec:
        from src import config
        model_spec = getattr(config, "fast_model", None)

    model = select_model(model_spec=model_spec)
    prompt = ROUTER_PROMPT.format(question=question)

    # 直接用 client 调用，限制 max_tokens 防止乱输出
    response = model.client.chat.completions.create(
        model=model.model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0,
    )
    result = response.choices[0].message.content.strip().lower()

    # 从返回中提取有效类别（LLM 可能返回多余文字）
    for cat in VALID_CATEGORIES:
        if cat in result:
            return cat

    # 无法识别时默认走 simple
    logger.warning(f"Router unrecognized response: '{result}', falling back to 'simple'")
    return "simple"


def get_lightrag_mode(category: str) -> str | None:
    """根据分类结果返回 LightRAG 查询 mode，chat 返回 None。"""
    return ROUTE_TO_MODE.get(category)
