from dataclasses import dataclass, field
from typing import Annotated

from src.agents.common.context import BaseContext
from src.agents.common.mcp import MCP_SERVERS

# @dataclass会自动帮你生成 __init__、__repr__、__eq__ 等常用方法，不用手动写。
# kw_only=True参数会强制该数据类的实例化必须使用关键字参数
@dataclass(kw_only=True)
class Context(BaseContext):
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="siliconflow/Qwen/Qwen3-235B-A22B-Instruct-2507",
        metadata={"name": "智能体模型", "options": [], "description": "智能体的驱动模型"},
    )

    mcps: list[str] = field(
        default_factory=list,
        metadata={"name": "MCP服务器", "options": list(MCP_SERVERS.keys()), "description": "MCP服务器列表"},
    )

    museum: str = field(
        default="",
        metadata={
            "name": "当前博物馆",
            "options": ["", "湖北省博物馆", "中国国家博物馆"],
            "description": "用户当前参观的博物馆，用于检索结果分层：优先返回本馆文物，其他馆作为扩展推荐。留空则不分层。",
        },
    )

    images: list = field(
        default_factory=list,
        metadata={
            "name": "用户上传图片",
            "description": "当前轮次用户上传的图片路径列表，由 chat_router 注入，用于图片相似度搜索工具。",
        },
    )
