import re
from typing import Any, cast

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.runtime import Runtime

from src.agents.common.base import BaseAgent
from src.agents.common.mcp import get_mcp_tools
from src.agents.common.models import load_chat_model
from src.utils import logger

from .context import Context
from .router import classify_query, get_lightrag_mode
from .state import State
from .tools import get_tools


class ChatbotAgent(BaseAgent):
    name = "智能体助手"
    description = "博物馆AI助手，自动使用知识库工具回答文物相关问题。"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph = None
        self.checkpointer = None
        self.context_schema = Context
        self.agent_tools = None

    def get_tools(self):
        # 获取所有工具
        return get_tools()

    # 只使用全馆1359条的两个主力库，其余历史实验库不暴露给 LLM
    ENABLED_KB_TOOLS = {
        "query_kb_08e43",  # 全馆文物知识图谱-1359 (LightRAG)
        "query_kb_d6f09",  # 全馆文物向量库-1359 (ChromaDB)
    }

    async def _get_all_tools(self, selected_mcps: list[str] = None):
        """获取可用工具：只保留主力知识库 + 非知识库工具（图片搜索、知识图谱等）。"""
        self.agent_tools = self.agent_tools or self.get_tools()

        all_tools = []
        for tool in self.agent_tools:
            if tool.name.startswith("query_kb_"):
                # 知识库工具：只保留主力库
                if tool.name in self.ENABLED_KB_TOOLS:
                    all_tools.append(tool)
            else:
                # 非知识库工具：全部保留（图片搜索、知识图谱、计算器等）
                all_tools.append(tool)

        if selected_mcps and isinstance(selected_mcps, list) and len(selected_mcps) > 0:
            for mcp in selected_mcps:
                all_tools.extend(await get_mcp_tools(mcp))

        return all_tools

    # ── 节点 1: router ──
    async def router_node(self, state: State, runtime: Runtime[Context]) -> dict:
        """路由节点：用轻量 LLM 对问题分类，决定检索策略。"""
        last_message = state.messages[-1]
        question = last_message.content if hasattr(last_message, "content") else str(last_message)

        query_type = classify_query(question)
        logger.info(f"Router: '{question[:50]}...' → {query_type}")
        return {"query_type": query_type}

    # ── 节点 2: chatbot（绑工具，LLM 决定是否调用）──
    async def llm_call(self, state: State, runtime: Runtime[Context] = None) -> dict[str, Any]:
        """LLM 节点：绑定所有工具，让 LLM 自主决定是否调用。"""
        model = load_chat_model(runtime.context.model)

        available_tools = await self._get_all_tools(runtime.context.mcps)

        # 用户上传了图片时，只绑定图片搜索工具，避免 LLM 多轮调用其他工具
        images = getattr(runtime.context, "images", [])
        if images:
            available_tools = [t for t in available_tools if t.name == "query_image_similarity"]
            logger.info(f"Image detected, restricted to image tool only: {[t.name for t in available_tools]}")
        else:
            logger.info(f"LLM binded ({len(available_tools)}) available_tools: {[tool.name for tool in available_tools]}")

        if available_tools:
            model = model.bind_tools(available_tools)

        # 动态构建 system prompt，注入当前博物馆信息
        system_prompt = runtime.context.system_prompt
        system_prompt += "\n\n重要：当工具返回的结果中包含图片（markdown格式如 ![文物名](图片URL)），你必须在回答中原样保留这些图片markdown语法，不要将它们改写为纯文本。用户需要看到图片。"
        museum = getattr(runtime.context, "museum", "")
        if museum:
            system_prompt += f"\n\n当前用户正在参观「{museum}」，请优先介绍该馆的文物，其他博物馆的相关文物可作为扩展推荐。"

        response = cast(
            AIMessage,
            await model.ainvoke([{"role": "system", "content": system_prompt}, *state.messages]),
        )

        # 后处理：确保 LLM 回复中提到的文物对应图片不丢失
        # 只补 LLM 回复中明确提到了文物名、但图片 URL 缺失的
        if response.content and not response.tool_calls:
            tool_images = {}
            for msg in state.messages:
                if isinstance(msg, ToolMessage):
                    for m in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)', msg.content):
                        name, url = m.group(1), m.group(2)
                        if name and len(name) >= 3 and name not in tool_images:
                            tool_images[name] = url
            if tool_images:
                missing = [(n, u) for n, u in tool_images.items()
                           if n in response.content and u not in response.content]
                if missing:
                    img_block = "\n\n"
                    for name, url in missing:
                        img_block += f"![{name}]({url})\n\n"
                    response.content += img_block
                    logger.info(f"Injected {len(missing)} missing images into LLM response")

        return {"messages": [response]}

    @staticmethod
    def _reorder_by_museum(result_text: str, museum: str) -> str:
        """将检索结果按博物馆标签重排序，本馆文物排前面。"""
        if not museum or not result_text:
            return result_text

        # LightRAG 返回的是纯文本，按分隔符拆分成条目
        # 常见分隔符：换行+横线、双换行等
        separators = ["\n---\n", "\n\n---\n\n", "\n-----\n"]
        sep_used = None
        for sep in separators:
            if sep in result_text:
                sep_used = sep
                break

        if not sep_used:
            # 没有明显分隔符，无法拆分排序，直接返回
            return result_text

        entries = result_text.split(sep_used)
        local_entries = []
        other_entries = []

        for entry in entries:
            if museum in entry:
                local_entries.append(entry)
            else:
                other_entries.append(entry)

        if not local_entries or not other_entries:
            # 全是本馆或全是外馆，不需要重排
            return result_text

        logger.info(f"Museum reorder: {len(local_entries)} local + {len(other_entries)} other entries")
        return sep_used.join(local_entries + other_entries)

    # ── 图片补充：从图片 collection 中按文物名精确匹配图片 URL ──
    @staticmethod
    def _get_image_name_map() -> dict:
        """获取或缓存 文物名→image_url 映射"""
        if hasattr(ChatbotAgent, "_image_name_map"):
            return ChatbotAgent._image_name_map

        import chromadb
        from chromadb.config import Settings
        from pathlib import Path
        from src import config as app_config

        CHROMA_DB_PATH = Path(getattr(app_config, "save_dir", "saves")) / "knowledge_base_data" / "chroma_data" / "chromadb"
        IMAGE_COLLECTION = "kb_d6f0936fffbeceb73dcd78a442dad8cb_images"

        try:
            client = chromadb.PersistentClient(
                path=str(CHROMA_DB_PATH),
                settings=Settings(anonymized_telemetry=False),
            )
            col = client.get_collection(name=IMAGE_COLLECTION)
            all_meta = col.get(include=["metadatas"])
            name_map = {}
            for meta in all_meta["metadatas"]:
                name = meta.get("name", "")
                url = meta.get("image_url", "")
                if name and url and name not in name_map:
                    name_map[name] = url
            ChatbotAgent._image_name_map = name_map
            logger.info(f"Loaded image name map: {len(name_map)} artifacts")
        except Exception as e:
            logger.warning(f"Failed to build image name map: {e}")
            ChatbotAgent._image_name_map = {}

        return ChatbotAgent._image_name_map

    @staticmethod
    def _fuzzy_match_image(artifact_name: str, name_map: dict) -> tuple[str, str] | None:
        """模糊匹配文物名到图片 URL。

        匹配策略（按优先级）：
        1. 精确匹配
        2. 图片库名包含查询名（如 '新春大吉' 匹配 '年画《新春大吉》'）
        3. 查询名包含图片库名（如 '彩绘人物车马出行图圆奁' 匹配 '彩绘人物车马出行图'）
        要求查询名至少 3 个字符，避免 '铜'、'玉' 这种短词误匹配。
        """
        if not artifact_name or len(artifact_name) < 3:
            return None

        # 1. 精确匹配
        if artifact_name in name_map:
            return (artifact_name, name_map[artifact_name])

        # 2. 图片库名包含查询名（查询名是核心词）
        best_match = None
        best_len = float('inf')
        for db_name, url in name_map.items():
            if artifact_name in db_name:
                # 选最短的匹配（最精确）
                if len(db_name) < best_len:
                    best_match = (db_name, url)
                    best_len = len(db_name)

        if best_match:
            return best_match

        # 3. 查询名包含图片库名（图片库名是核心词，至少3字）
        best_match = None
        best_len = 0
        for db_name, url in name_map.items():
            if len(db_name) >= 3 and db_name in artifact_name:
                # 选最长的匹配（最精确）
                if len(db_name) > best_len:
                    best_match = (db_name, url)
                    best_len = len(db_name)

        return best_match

    @staticmethod
    def _enrich_result_with_images(result: Any) -> str:
        """对知识库工具结果，从每条结果中提取文物名，模糊匹配图片后内联插入。"""
        name_map = ChatbotAgent._get_image_name_map()
        if not name_map:
            return str(result)

        # ChromaDB 返回 list[dict]，每条有 content/metadata/score
        if isinstance(result, list) and result and isinstance(result[0], dict) and "content" in result[0]:
            enriched_items = []
            matched_count = 0
            for item in result:
                content = item.get("content", "")
                first_line = content.split("\n")[0].strip().lstrip("# ")
                match = ChatbotAgent._fuzzy_match_image(first_line, name_map)
                item_text = str(item)
                if match:
                    db_name, image_url = match
                    item_text += f"\n![{first_line}]({image_url})"
                    matched_count += 1
                enriched_items.append(item_text)
            enriched_text = "\n".join(enriched_items)
            if matched_count:
                logger.info(f"Enriched {matched_count}/{len(result)} ChromaDB results with images")
            return enriched_text

        # LightRAG 或其他纯文本结果
        result_text = str(result)
        lines = result_text.split("\n")
        matched_images = []
        seen_urls = set()
        for line in lines:
            name = line.strip().lstrip("# ").strip("\"'《》")
            if not name or len(name) < 3:
                continue
            match = ChatbotAgent._fuzzy_match_image(name, name_map)
            if match and match[1] not in seen_urls:
                matched_images.append((name, match[1]))
                seen_urls.add(match[1])

        if matched_images:
            img_section = "\n\n"
            for name, url in matched_images:
                img_section += f"![{name}]({url})\n"
            result_text += img_section
            logger.info(f"Enriched LightRAG result with {len(matched_images)} images")

        return result_text

    # ── 节点 3: tools（执行工具，传递 mode）──
    async def dynamic_tools_node(self, state: State, runtime: Runtime[Context]) -> dict[str, list[ToolMessage]]:
        """工具节点：执行 LLM 请求的工具调用，并根据 router 分类传递 LightRAG mode。"""
        available_tools = await self._get_all_tools(runtime.context.mcps)
        tools_by_name = {t.name: t for t in available_tools}

        # 根据 router 分类确定 LightRAG 检索模式（chat 类型返回 None，不注入 mode）
        mode = get_lightrag_mode(state.query_type)
        museum = getattr(runtime.context, "museum", "")
        logger.info(f"Tools node: query_type={state.query_type}, lightrag_mode={mode}, museum={museum}")

        last_message = state.messages[-1]
        results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"].copy()

            # 对知识库工具注入 mode 参数（chat 类型不注入，使用知识库默认 mode）
            if tool_name.startswith("query_") and mode is not None:
                tool_args["mode"] = mode

            # 对图片搜索工具注入用户上传的图片路径
            images = getattr(runtime.context, "images", [])
            if tool_name == "query_image_similarity" and images and not tool_args.get("query_img"):
                tool_args["query_img"] = images[0]  # 使用第一张上传的图片

            tool = tools_by_name.get(tool_name)
            if not tool:
                results.append(ToolMessage(content=f"工具 {tool_name} 不存在", tool_call_id=tool_call["id"]))
                continue

            try:
                result = await tool.ainvoke(tool_args)

                # 对非图片搜索的知识库工具，在 str 转换前精确匹配文物图片
                if tool_name.startswith("query_") and tool_name != "query_image_similarity":
                    try:
                        result_text = self._enrich_result_with_images(result)
                    except Exception as e:
                        logger.warning(f"Image enrichment failed: {e}")
                        result_text = str(result)
                else:
                    result_text = str(result)

                # 按博物馆标签重排序：本馆文物排前面
                if museum and tool_name.startswith("query_"):
                    result_text = self._reorder_by_museum(result_text, museum)

                results.append(ToolMessage(content=result_text, tool_call_id=tool_call["id"]))
            except Exception as e:
                logger.error(f"Tool {tool_name} error: {e}")
                results.append(ToolMessage(content=f"工具调用失败: {e}", tool_call_id=tool_call["id"]))

        return {"messages": results}

    async def get_graph(self, **kwargs):
        """构建图：
        START → router → chatbot ←→ tools → END
        router 分类结果写入 state.query_type，tools 节点据此注入 LightRAG mode。
        """
        if self.graph:
            return self.graph

        builder = StateGraph(State, context_schema=self.context_schema)

        # 添加节点
        builder.add_node("router", self.router_node)
        builder.add_node("chatbot", self.llm_call)
        builder.add_node("tools", self.dynamic_tools_node)

        # 连线: router → chatbot ←→ tools → END
        builder.add_edge(START, "router")
        builder.add_edge("router", "chatbot")
        builder.add_conditional_edges("chatbot", tools_condition)
        builder.add_edge("tools", "chatbot")

        self.checkpointer = await self._get_checkpointer()
        graph = builder.compile(checkpointer=self.checkpointer, name=self.name)
        self.graph = graph
        return graph


def main():
    pass


if __name__ == "__main__":
    main()
    # asyncio.run(main())
