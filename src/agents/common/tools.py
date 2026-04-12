import asyncio
import os
import traceback
from typing import Annotated, Any

from langchain_core.tools import StructuredTool, tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from src import config, graph_base, knowledge_base
from src.utils import logger


@tool
def query_knowledge_graph(query: Annotated[str, "The keyword to query knowledge graph."]) -> Any:
    """Use this to query knowledge graph, which include some food domain knowledge."""
    try:
        logger.debug(f"Querying knowledge graph with: {query}")
        result = graph_base.query_node(query, hops=2, return_format="triples")
        logger.debug(
            f"Knowledge graph query returned "
            f"{len(result.get('triples', [])) if isinstance(result, dict) else 'N/A'} triples"
        )
        return result
    except Exception as e:
        logger.error(f"Knowledge graph query error: {e}, {traceback.format_exc()}")
        return f"知识图谱查询失败: {str(e)}"


def get_static_tools() -> list:
    """注册静态工具"""
    static_tools = [
        query_knowledge_graph,
    ]

    # 检查是否启用网页搜索
    if config.enable_web_search:
        static_tools.append(TavilySearch(max_results=10))

    return static_tools


class KnowledgeRetrieverModel(BaseModel):
    model_config = {"extra": "allow"}

    query_text: str = Field(
        default="",
        description=(
            "当用户提供的输入中包含关键词时，请提供一个查询的关键词，查询的时候，应该尽量以可能帮助回答这个问题的关键词进行查询，不要直接使用用户的原始输入去查询。如果没有请忽略这个字段。"
        )
    )
    query_img: str = Field(
        default="",
        description=(
            "当用户提供的输入中包含图片url时，则请提供图片的URL去查询,否则请忽略这个字段。"
        )
    )
    query_desc: str = Field(
        default="",
        description=(
            "当用户提供的输入中包含文物的材质类型、外观、纹样、图案等信息描述时，则请提供文物的描述去查询,否则请忽略这个字段。"
        )
    )


class ImageSearchModel(BaseModel):
    """图片相似度搜索工具的参数模型"""
    model_config = {"extra": "allow"}

    query_img: str = Field(
        default="",
        description="用户上传的图片URL或本地路径。当用户发送了图片并想查找相似文物时使用。",
    )
    query_desc: str = Field(
        default="",
        description="文物的外观描述文本（如材质、纹样、形状等）。用CN-CLIP文本编码器在图片向量空间中检索相似文物。",
    )


def _create_image_search_tool():
    """创建图片相似度搜索工具，直接查询 ChromaDB 图片 collection"""
    import chromadb
    from chromadb.config import Settings
    from pathlib import Path

    CHROMA_DB_PATH = Path(getattr(config, "save_dir", "saves")) / "knowledge_base_data" / "chroma_data" / "chromadb"
    # 默认使用全馆文物向量库的图片 collection
    DEFAULT_IMAGE_DB_ID = "kb_d6f0936fffbeceb73dcd78a442dad8cb"

    async def query_image_similarity(query_img: str = "", query_desc: str = "", **kwargs) -> Any:
        """通过图片或文物描述，在博物馆文物图片库中搜索视觉相似的文物。"""
        import asyncio

        def _do_query():
            from src.knowledge.utils.image_embedding_utils import get_image_embedding, get_text_embedding

            db_id = kwargs.get("db_id", DEFAULT_IMAGE_DB_ID)
            collection_name = f"{db_id}_images"
            top_k = kwargs.get("top_k", 5)

            try:
                client = chromadb.PersistentClient(
                    path=str(CHROMA_DB_PATH),
                    settings=Settings(anonymized_telemetry=False),
                )
                collection = client.get_collection(name=collection_name)
            except Exception as e:
                logger.error(f"Image collection not found: {collection_name}, error: {e}")
                return f"图片向量库未找到: {collection_name}"

            results = []

            # 图片相似度搜索
            if query_img:
                try:
                    embedding = get_image_embedding(query_img)
                    if embedding is not None and len(embedding) > 0:
                        image_results = collection.query(
                            query_embeddings=[embedding.tolist() if hasattr(embedding, 'tolist') else embedding],
                            n_results=top_k,
                            include=["documents", "metadatas", "distances"],
                        )
                        if image_results and image_results.get("documents"):
                            for i, doc in enumerate(image_results["documents"][0]):
                                if doc:
                                    meta = image_results["metadatas"][0][i] if image_results.get("metadatas") else {}
                                    score = 1 - image_results["distances"][0][i] if image_results.get("distances") else 0.0
                                    results.append({
                                        "content": doc,
                                        "metadata": meta,
                                        "score": round(score, 4),
                                        "match_type": "图片相似",
                                    })
                except Exception as e:
                    logger.error(f"Image embedding query failed: {e}")

            # 描述文本相似度搜索（用 CN-CLIP 文本编码器在图片向量空间检索）
            if query_desc:
                try:
                    embedding = get_text_embedding(query_desc)
                    if embedding is not None and len(embedding) > 0:
                        desc_results = collection.query(
                            query_embeddings=[embedding.tolist() if hasattr(embedding, 'tolist') else embedding],
                            n_results=top_k,
                            include=["documents", "metadatas", "distances"],
                        )
                        if desc_results and desc_results.get("documents"):
                            for i, doc in enumerate(desc_results["documents"][0]):
                                if doc:
                                    meta = desc_results["metadatas"][0][i] if desc_results.get("metadatas") else {}
                                    score = 1 - desc_results["distances"][0][i] if desc_results.get("distances") else 0.0
                                    results.append({
                                        "content": doc,
                                        "metadata": meta,
                                        "score": round(score, 4),
                                        "match_type": "描述相似",
                                    })
                except Exception as e:
                    logger.error(f"Text embedding query failed: {e}")

            if not results:
                return "未找到相似文物。请尝试提供更清晰的图片或更详细的描述。"

            # 按 score 降序排列，去重
            seen_names = set()
            unique_results = []
            for r in sorted(results, key=lambda x: x["score"], reverse=True):
                name = r["metadata"].get("name", "")
                if name not in seen_names:
                    seen_names.add(name)
                    unique_results.append(r)

            # 格式化为包含图片的文本，方便 LLM 在回答中展示图片
            formatted = []
            for i, r in enumerate(unique_results[:top_k]):
                meta = r["metadata"]
                name = meta.get("name", "未知")
                museum = meta.get("museum", "未知")
                score = r["score"]
                # 优先使用远程原始图片链接，本地路径作为备选
                image_url = meta.get("image_url", "")
                if not image_url:
                    local_path = meta.get("local_path", "")
                    if local_path:
                        filename = os.path.basename(local_path)
                        image_url = f"/static/museum_images/{filename}"

                item = f"【{i+1}】{name}（{museum}，相似度: {score}）\n"
                item += f"  描述: {r['content'][:200]}\n"
                if image_url:
                    item += f"  ![{name}]({image_url})\n"
                formatted.append(item)

            return "\n".join(formatted)

        return await asyncio.to_thread(_do_query)

    tool = StructuredTool.from_function(
        coroutine=query_image_similarity,
        name="query_image_similarity",
        description=(
            "通过图片或文物外观描述搜索相似文物。适用场景：\n"
            "1. 用户上传了一张文物图片，想知道这是什么文物或找相似文物\n"
            "2. 用户用文字描述了文物的外观特征（材质、纹样、形状等），想找匹配的文物\n"
            "注意：这个工具基于视觉特征搜索，与文本知识库检索互补。\n"
            "返回结果包含图片URL，回答时请用 markdown 图片语法展示：![文物名](图片URL)"
        ),
        args_schema=ImageSearchModel,
        metadata={"name": "图片相似度搜索", "tag": ["knowledgebase", "image"]},
    )
    return tool


def get_kb_based_tools() -> list:
    """获取所有知识库基于的工具"""
    # 获取所有知识库
    kb_tools = []
    retrievers = knowledge_base.get_retrievers()

    def _create_retriever_wrapper(db_id: str, retriever_info: dict[str, Any]):
        """创建检索器包装函数的工厂函数，避免闭包变量捕获问题"""

        async def async_retriever_wrapper(query_text: str = "", query_img: str = "", query_desc: str = "", **kwargs) -> Any:
            """异步检索器包装函数"""
            retriever = retriever_info["retriever"]
            try:
                logger.debug(f"Retrieving from database {db_id} with query: {query_text}, query_img: {query_img}, query_desc: {query_desc}, kwargs: {kwargs}")
                if asyncio.iscoroutinefunction(retriever):
                    result = await retriever(query_text, query_img, query_desc, **kwargs)
                else:
                    result = retriever(query_text, query_img, query_desc, **kwargs)
                logger.debug(f"Retrieved {len(result) if isinstance(result, list) else 'N/A'} results from {db_id}")
                return result
            except Exception as e:
                logger.error(f"Error in retriever {db_id}: {e}")
                return f"检索失败: {str(e)}"

        return async_retriever_wrapper

    for db_id, retrieve_info in retrievers.items():
        try:
            # 使用改进的工具ID生成策略
            tool_id = f"query_{db_id[:8]}"

            # 构建工具描述
            description = (
                f"使用 {retrieve_info['name']} 知识库进行检索。\n"
                f"下面是这个知识库的描述：\n{retrieve_info['description'] or '没有描述。'} "
            )

            # 使用工厂函数创建检索器包装函数，避免闭包问题
            retriever_wrapper = _create_retriever_wrapper(db_id, retrieve_info)

            # 使用 StructuredTool.from_function 创建异步工具
            tool = StructuredTool.from_function(
                coroutine=retriever_wrapper,
                name=tool_id,
                description=description,
                args_schema=KnowledgeRetrieverModel,
                metadata=retrieve_info["metadata"] | {"tag": ["knowledgebase"]},
            )

            kb_tools.append(tool)
            # logger.debug(f"Successfully created tool {tool_id} for database {db_id}")

        except Exception as e:
            logger.error(f"Failed to create tool for database {db_id}: {e}, \n{traceback.format_exc()}")
            continue

    return kb_tools


def get_buildin_tools() -> list:
    """获取所有可运行的工具（给大模型使用）"""
    tools = []

    try:
        # 获取所有知识库基于的工具
        tools.extend(get_kb_based_tools())
        tools.extend(get_static_tools())

        # 添加图片相似度搜索工具
        try:
            tools.append(_create_image_search_tool())
            logger.info("Image similarity search tool registered")
        except Exception as e:
            logger.warning(f"Failed to create image search tool: {e}")

        # from src.agents.common.toolkits.mysql.tools import get_mysql_tools

        # tools.extend(get_mysql_tools())

    except Exception as e:
        logger.error(f"Failed to get knowledge base retrievers: {e}")

    return tools


def gen_tool_info(tools) -> list[dict[str, Any]]:
    """获取所有工具的信息（用于前端展示）"""
    tools_info = []

    try:
        # 获取注册的工具信息
        for tool_obj in tools:
            try:
                metadata = getattr(tool_obj, "metadata", {}) or {}
                info = {
                    "id": tool_obj.name,
                    "name": metadata.get("name", tool_obj.name),
                    "description": tool_obj.description,
                    "metadata": metadata,
                    "args": [],
                    # "is_async": is_async  # Include async information
                }

                if hasattr(tool_obj, "args_schema") and tool_obj.args_schema:
                    schema = tool_obj.args_schema.schema()
                    for arg_name, arg_info in schema.get("properties", {}).items():
                        info["args"].append(
                            {
                                "name": arg_name,
                                "type": arg_info.get("type", ""),
                                "description": arg_info.get("description", ""),
                            }
                        )

                tools_info.append(info)
                # logger.debug(f"Successfully processed tool info for {tool_obj.name}")

            except Exception as e:
                logger.error(
                    f"Failed to process tool {getattr(tool_obj, 'name', 'unknown')}: {e}\n{traceback.format_exc()}"
                )
                continue

    except Exception as e:
        logger.error(f"Failed to get tools info: {e}\n{traceback.format_exc()}")
        return []

    logger.info(f"Successfully extracted info for {len(tools_info)} tools")
    return tools_info
