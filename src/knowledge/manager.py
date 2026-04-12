import asyncio
import json
import os
import re

from src import config
from src.knowledge.base import KBNotFoundError, KnowledgeBase
from src.knowledge.factory import KnowledgeBaseFactory
from src.models.rerank import get_reranker
from src.utils import logger
from src.utils.datetime_utils import coerce_any_to_utc_datetime, utc_isoformat


class KnowledgeBaseManager:
    """
    知识库管理器

    统一管理多种类型的知识库实例，提供统一的外部接口
    """

    def __init__(self, work_dir: str):
        """
        初始化知识库管理器

        Args:
            work_dir: 工作目录
        """
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

        # 知识库实例缓存 {kb_type: kb_instance}
        self.kb_instances: dict[str, KnowledgeBase] = {}

        # 全局数据库元信息 {db_id: metadata_with_kb_type}
        self.global_databases_meta: dict[str, dict] = {}

        # 元数据锁
        self._metadata_lock = asyncio.Lock()

        # 加载全局元数据
        self._load_global_metadata()
        self._normalize_global_metadata()

        # 初始化已存在的知识库实例
        self._initialize_existing_kbs()

        logger.info("KnowledgeBaseManager initialized")

    def _load_global_metadata(self):
        """加载全局元数据"""
        meta_file = os.path.join(self.work_dir, "global_metadata.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, encoding="utf-8") as f:
                    data = json.load(f)
                    self.global_databases_meta = data.get("databases", {})
                logger.info(f"Loaded global metadata for {len(self.global_databases_meta)} databases")
            except Exception as e:
                logger.error(f"Failed to load global metadata: {e}")

    def _save_global_metadata(self):
        """保存全局元数据"""
        meta_file = os.path.join(self.work_dir, "global_metadata.json")
        data = {"databases": self.global_databases_meta, "updated_at": utc_isoformat(), "version": "2.0"}
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _normalize_global_metadata(self) -> None:
        """Normalize stored timestamps within the global metadata cache."""
        for meta in self.global_databases_meta.values():
            if "created_at" in meta:
                try:
                    dt_value = coerce_any_to_utc_datetime(meta.get("created_at"))
                    if dt_value:
                        meta["created_at"] = utc_isoformat(dt_value)
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Failed to normalize database metadata timestamp {meta.get('created_at')!r}: {exc}")

    def _initialize_existing_kbs(self):
        """初始化已存在的知识库实例"""
        kb_types_in_use = set()
        for db_meta in self.global_databases_meta.values():
            kb_type = db_meta.get("kb_type", "lightrag")  # 默认为lightrag
            kb_types_in_use.add(kb_type)

        # 为每种使用中的知识库类型创建实例
        for kb_type in kb_types_in_use:
            try:
                self._get_or_create_kb_instance(kb_type)
            except Exception as e:
                logger.error(f"Failed to initialize {kb_type} knowledge base: {e}")

    def _get_or_create_kb_instance(self, kb_type: str) -> KnowledgeBase:
        """
        获取或创建知识库实例

        Args:
            kb_type: 知识库类型

        Returns:
            知识库实例
        """
        if kb_type in self.kb_instances:
            return self.kb_instances[kb_type]

        # 创建新的知识库实例
        kb_work_dir = os.path.join(self.work_dir, f"{kb_type}_data")
        kb_instance = KnowledgeBaseFactory.create(kb_type, kb_work_dir)

        self.kb_instances[kb_type] = kb_instance
        logger.info(f"Created {kb_type} knowledge base instance")
        return kb_instance

    def _get_kb_for_database(self, db_id: str) -> KnowledgeBase:
        """
        根据数据库ID获取对应的知识库实例

        Args:
            db_id: 数据库ID

        Returns:
            知识库实例

        Raises:
            KBNotFoundError: 数据库不存在或知识库类型不支持
        """
        if db_id not in self.global_databases_meta:
            raise KBNotFoundError(f"Database {db_id} not found")

        kb_type = self.global_databases_meta[db_id].get("kb_type", "lightrag")

        if not KnowledgeBaseFactory.is_type_supported(kb_type):
            raise KBNotFoundError(f"Unsupported knowledge base type: {kb_type}")

        return self._get_or_create_kb_instance(kb_type)

    # =============================================================================
    # 统一的外部接口 - 与原始 LightRagBasedKB 兼容
    # =============================================================================

    def get_kb(self, db_id: str) -> KnowledgeBase:
        """Public accessor to fetch the underlying knowledge base instance by database id.

        This provides a simple compatibility layer for callers that expect a
        `get_kb` method on the manager.
        """
        return self._get_kb_for_database(db_id)

    def get_databases(self) -> dict:
        """获取所有数据库信息"""
        all_databases = []

        # 收集所有知识库的数据库信息
        for kb_type, kb_instance in self.kb_instances.items():
            kb_databases = kb_instance.get_databases()["databases"]
            all_databases.extend(kb_databases)

        return {"databases": all_databases}

    async def create_database(
        self, database_name: str, description: str, kb_type: str, embed_info: dict | None = None, **kwargs
    ) -> dict:
        """
        创建数据库

        Args:
            database_name: 数据库名称
            description: 数据库描述
            kb_type: 知识库类型，默认为lightrag
            embed_info: 嵌入模型信息
            **kwargs: 其他配置参数，包括chunk_size和chunk_overlap

        Returns:
            数据库信息字典
        """
        if not KnowledgeBaseFactory.is_type_supported(kb_type):
            available_types = list(KnowledgeBaseFactory.get_available_types().keys())
            raise ValueError(f"Unsupported knowledge base type: {kb_type}. Available types: {available_types}")

        kb_instance = self._get_or_create_kb_instance(kb_type)

        db_info = kb_instance.create_database(database_name, description, embed_info, **kwargs)
        db_id = db_info["db_id"]

        async with self._metadata_lock:
            self.global_databases_meta[db_id] = {
                "name": database_name,
                "description": description,
                "kb_type": kb_type,
                "created_at": utc_isoformat(),
                "additional_params": kwargs.copy(),
            }
            self._save_global_metadata()

        logger.info(f"Created {kb_type} database: {database_name} ({db_id}) with {kwargs}")
        return db_info

    async def delete_database(self, db_id: str) -> dict:
        """删除数据库"""
        try:
            kb_instance = self._get_kb_for_database(db_id)
            result = kb_instance.delete_database(db_id)

            async with self._metadata_lock:
                if db_id in self.global_databases_meta:
                    del self.global_databases_meta[db_id]
                    self._save_global_metadata()

            return result
        except KBNotFoundError as e:
            logger.warning(f"Database {db_id} not found during deletion: {e}")
            return {"message": "删除成功"}

    async def add_content(self, db_id: str, items: list[str], params: dict | None = None) -> list[dict]:
        """添加内容（文件/URL）"""
        kb_instance = self._get_kb_for_database(db_id)
        return await kb_instance.add_content(db_id, items, params or {})

    async def add_image_embeddings(self, db_id: str, items: list[str], params: dict | None = None) -> list[dict]:
        """添加图片嵌入"""
        kb_instance = self._get_kb_for_database(db_id)
        return await kb_instance.add_image_embeddings(db_id, items, params or {})

    async def aquery(self, query_text: str, db_id: str, **kwargs) -> list[dict]:
        """异步查询知识库"""
        # 提取 museum 参数（不传给底层 LightRAG，QueryParam 不认识它）
        target_museum = kwargs.pop("museum", "")

        kb_instance = self._get_kb_for_database(db_id)

        # 执行基础查询
        results = await kb_instance.aquery(db_id, query_text, **kwargs)

        # LightRAG 返回 str（only_need_context 模式），拆分成多条独立 chunk 以支持 rerank
        if isinstance(results, str):
            if results.strip():
                results = self._split_lightrag_chunks(self._tag_museum_source(results))
            else:
                results = []

        # 检查是否启用重排序功能
        if config.enable_reranker and results:
            try:
                # 获取重排序器实例
                reranker = get_reranker(config.reranker)
                
                # 准备重排序输入：查询文本和所有检索结果的文本内容
                sentences = [result["content"] for result in results]
                sentence_pairs = (query_text, sentences)
                
                # 计算重排序分数
                rerank_scores = reranker.compute_score(sentence_pairs, normalize=True)
                
                # 将重排序分数添加到结果中
                for i, result in enumerate(results):
                    if i < len(rerank_scores):
                        result["rerank_score"] = rerank_scores[i]
                    else:
                        result["rerank_score"] = 0.0
                
                logger.debug(f"Applied reranking to {len(results)} results")
                
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
                # 重排序失败时，为所有结果添加默认的重排序分数
                for result in results:
                    result["rerank_score"] = result.get("score", 0.0)
        
        # 按博物馆硬分层：本馆结果在前，其他馆标记为扩展推荐在后
        if target_museum and results:
            results = self._prioritize_by_museum(results, target_museum)

        return results

    @staticmethod
    def _prioritize_by_museum(results: list[dict], target_museum: str) -> list[dict]:
        """按博物馆分层排序：本馆结果在前，其他馆作为扩展推荐在后。

        知识图谱摘要（source=lightrag_kg）始终保留在最前面。
        """
        kg_results = []
        primary = []
        extended = []

        for item in results:
            source = item.get("metadata", {}).get("source", "")
            museum = item.get("metadata", {}).get("museum", "")

            if source == "lightrag_kg":
                # 知识图谱摘要始终保留
                kg_results.append(item)
            elif target_museum in museum or museum in target_museum:
                # 本馆结果
                primary.append(item)
            else:
                # 其他馆 → 标记为扩展推荐
                item.setdefault("metadata", {})["is_extended"] = True
                extended.append(item)

        logger.info(
            f"Museum filter '{target_museum}': "
            f"{len(primary)} primary, {len(extended)} extended, {len(kg_results)} kg"
        )
        return kg_results + primary + extended

    @staticmethod
    def _split_lightrag_chunks(tagged_text: str) -> list[dict]:
        """将 LightRAG 的聚合文本拆分成独立的 chunk 条目。

        从 Document Chunks 部分提取每条记录作为独立 dict，
        同时保留 Entity + Relationship 作为一条知识图谱摘要。
        这样 reranker 可以对每条 chunk 单独打分排序。
        """
        results = []

        # 提取 Document Chunks 中的每条 JSON
        for line in tagged_text.split('\n'):
            stripped = line.strip()
            if not (stripped.startswith('{') and stripped.endswith('}')):
                continue
            try:
                obj = json.loads(stripped)
                if "reference_id" in obj and "content" in obj:
                    results.append({
                        "content": obj["content"],
                        "score": 1.0,
                        "metadata": {
                            "source": "lightrag_chunk",
                            "museum": obj.get("museum", ""),
                            "reference_id": obj.get("reference_id", ""),
                        },
                    })
            except (json.JSONDecodeError, KeyError):
                continue

        # 将 Entity + Relationship 部分整合为一条知识图谱摘要（放在最前面）
        kg_lines = []
        for line in tagged_text.split('\n'):
            stripped = line.strip()
            if not (stripped.startswith('{') and stripped.endswith('}')):
                kg_lines.append(line)
                continue
            try:
                obj = json.loads(stripped)
                if "entity" in obj or "entity1" in obj:
                    kg_lines.append(line)
            except (json.JSONDecodeError, KeyError):
                kg_lines.append(line)

        kg_text = '\n'.join(kg_lines).strip()
        if kg_text:
            results.insert(0, {
                "content": kg_text,
                "score": 1.0,
                "metadata": {"source": "lightrag_kg"},
            })

        return results if results else [{"content": tagged_text, "score": 1.0, "metadata": {"source": "lightrag"}}]

    @staticmethod
    def _dedup_sep(text: str) -> str:
        """清理 <SEP> 分隔的重复内容，保留唯一片段"""
        if "<SEP>" not in text:
            return text
        parts = [p.strip() for p in text.split("<SEP>")]
        seen = []
        for p in parts:
            if p and p not in seen:
                seen.append(p)
        return "<SEP>".join(seen)

    @staticmethod
    def _tag_museum_source(text: str) -> str:
        """对 LightRAG 返回文本做后处理：添加博物馆标签 + 去除 <SEP> 重复。

        从 Document Chunks 中提取 文物名→收藏单位 的映射，
        然后回填到 Entity / Chunk 的 JSON 条目中，让下游 LLM 能区分每件文物属于哪个博物馆。
        """
        # Step 1: 从 chunk 内容中建立 文物名 → 博物馆 映射
        museum_map: dict[str, str] = {}
        for m in re.finditer(r'【(.+?)】.*?收藏单位[：:]\s*([^\n\\]+)', text):
            artifact_name = m.group(1).strip()
            museum_name = m.group(2).strip().rstrip('"')
            museum_map[artifact_name] = museum_name

        if not museum_map:
            return text

        def _find_museum(name: str) -> str | None:
            """根据文物名查找博物馆，支持精确和模糊匹配"""
            if name in museum_map:
                return museum_map[name]
            for artifact, museum in museum_map.items():
                if name in artifact or artifact in name:
                    return museum
            return None

        # Step 2: 逐行���理 — 注入 museum 字段 + 去除 description 中的 <SEP> 重复
        lines = text.split('\n')
        result_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('{') and stripped.endswith('}'):
                try:
                    obj = json.loads(stripped)
                    modified = False
                    # 注入博物馆标签
                    museum = None
                    if "entity" in obj:
                        museum = _find_museum(obj["entity"])
                    elif "content" in obj:
                        m = re.search(r'收藏单位[：:]\s*([^\n]+)', obj["content"])
                        if m:
                            museum = m.group(1).strip()
                    if museum:
                        obj["museum"] = museum
                        modified = True
                    # 去重 description 中的 <SEP>
                    if "description" in obj and "<SEP>" in obj["description"]:
                        obj["description"] = KnowledgeBaseManager._dedup_sep(obj["description"])
                        modified = True
                    if modified:
                        indent = line[:len(line) - len(line.lstrip())]
                        result_lines.append(indent + json.dumps(obj, ensure_ascii=False))
                        continue
                except (json.JSONDecodeError, KeyError):
                    pass
            result_lines.append(line)

        return '\n'.join(result_lines)

    async def export_data(self, db_id: str, format: str = "zip", **kwargs) -> str:
        """导出知识库数据"""
        kb_instance = self._get_kb_for_database(db_id)
        return await kb_instance.export_data(db_id, format=format, **kwargs)

    def query(self, query_text: str, db_id: str, **kwargs) -> str:
        """同步查询知识库（兼容性方法）"""
        kb_instance = self._get_kb_for_database(db_id)
        return kb_instance.query(query_text, db_id, **kwargs)

    def get_database_info(self, db_id: str) -> dict | None:
        """获取数据库详细信息"""
        try:
            kb_instance = self._get_kb_for_database(db_id)
            db_info = kb_instance.get_database_info(db_id)

            # 添加全局元数据中的additional_params信息
            if db_info and db_id in self.global_databases_meta:
                global_meta = self.global_databases_meta[db_id]
                additional_params = global_meta.get("additional_params", {})
                if additional_params:
                    db_info["additional_params"] = additional_params

            return db_info
        except KBNotFoundError:
            return None

    async def delete_file(self, db_id: str, file_id: str) -> None:
        """删除文件"""
        kb_instance = self._get_kb_for_database(db_id)
        await kb_instance.delete_file(db_id, file_id)

    async def get_file_basic_info(self, db_id: str, file_id: str) -> dict:
        """获取文件基本信息（仅元数据）"""
        kb_instance = self._get_kb_for_database(db_id)
        return await kb_instance.get_file_basic_info(db_id, file_id)

    async def get_file_content(self, db_id: str, file_id: str) -> dict:
        """获取文件内容信息（chunks和lines）"""
        kb_instance = self._get_kb_for_database(db_id)
        return await kb_instance.get_file_content(db_id, file_id)

    async def get_file_info(self, db_id: str, file_id: str) -> dict:
        """获取文件完整信息（基本信息+内容信息）- 保持向后兼容"""
        kb_instance = self._get_kb_for_database(db_id)
        return await kb_instance.get_file_info(db_id, file_id)

    def get_db_upload_path(self, db_id: str | None = None) -> str:
        """获取数据库上传路径"""
        if db_id:
            try:
                kb_instance = self._get_kb_for_database(db_id)
                return kb_instance.get_db_upload_path(db_id)
            except KBNotFoundError:
                # 如果数据库不存在，创建通用上传路径
                pass

        # 通用上传路径
        general_uploads = os.path.join(self.work_dir, "uploads")
        os.makedirs(general_uploads, exist_ok=True)
        return general_uploads

    def file_existed_in_db(self, db_id: str | None, content_hash: str | None) -> bool:
        """检查指定数据库中是否存在相同内容哈希的文件"""
        if not db_id or not content_hash:
            return False

        try:
            kb_instance = self._get_kb_for_database(db_id)
        except KBNotFoundError:
            return False

        for file_info in kb_instance.files_meta.values():
            if file_info.get("database_id") != db_id:
                continue
            if file_info.get("content_hash") == content_hash:
                return True

        return False

    async def update_database(self, db_id: str, name: str, description: str) -> dict:
        """更新数据库"""
        kb_instance = self._get_kb_for_database(db_id)
        result = kb_instance.update_database(db_id, name, description)

        async with self._metadata_lock:
            if db_id in self.global_databases_meta:
                self.global_databases_meta[db_id]["name"] = name
                self.global_databases_meta[db_id]["description"] = description
                self._save_global_metadata()

        return result

    def get_retrievers(self) -> dict[str, dict]:
        """获取所有检索器"""
        all_retrievers = {}

        # 收集所有知识库的检索器
        for kb_instance in self.kb_instances.values():
            retrievers = kb_instance.get_retrievers()
            all_retrievers.update(retrievers)

        return all_retrievers

    # =============================================================================
    # 管理器特有的方法
    # =============================================================================

    def get_supported_kb_types(self) -> dict[str, dict]:
        """获取支持的知识库类型"""
        return KnowledgeBaseFactory.get_available_types()

    def get_kb_instance_info(self) -> dict[str, dict]:
        """获取知识库实例信息"""
        info = {}
        for kb_type, kb_instance in self.kb_instances.items():
            info[kb_type] = {
                "work_dir": kb_instance.work_dir,
                "database_count": len(kb_instance.databases_meta),
                "file_count": len(kb_instance.files_meta),
            }
        return info

    def get_statistics(self) -> dict:
        """获取统计信息"""
        stats = {"total_databases": len(self.global_databases_meta), "kb_types": {}, "total_files": 0}

        # 按知识库类型统计
        for db_meta in self.global_databases_meta.values():
            kb_type = db_meta.get("kb_type", "lightrag")
            if kb_type not in stats["kb_types"]:
                stats["kb_types"][kb_type] = 0
            stats["kb_types"][kb_type] += 1

        # 统计文件总数
        for kb_instance in self.kb_instances.values():
            stats["total_files"] += len(kb_instance.files_meta)

        return stats

    # =============================================================================
    # 兼容性方法 - 为了支持现有的 graph_router.py
    # =============================================================================

    async def _get_lightrag_instance(self, db_id: str):
        """
        获取 LightRAG 实例（兼容性方法）

        Args:
            db_id: 数据库ID

        Returns:
            LightRAG 实例，如果数据库不是 lightrag 类型则返回 None

        Raises:
            ValueError: 如果数据库不存在或不是 lightrag 类型
        """
        try:
            # 检查数据库是否存在
            if db_id not in self.global_databases_meta:
                logger.error(f"Database {db_id} not found in global metadata")
                return None

            # 检查是否是 LightRAG 类型
            kb_type = self.global_databases_meta[db_id].get("kb_type", "lightrag")
            if kb_type != "lightrag":
                logger.error(f"Database {db_id} is not a LightRAG type (actual type: {kb_type})")
                raise ValueError(f"Database {db_id} is not a LightRAG knowledge base")

            # 获取 LightRAG 知识库实例
            kb_instance = self._get_kb_for_database(db_id)

            # 如果不是 LightRagKB 实例，返回错误
            if not hasattr(kb_instance, "_get_lightrag_instance"):
                logger.error(f"Knowledge base instance for {db_id} is not LightRagKB")
                return None

            # 调用 LightRagKB 的方法获取 LightRAG 实例
            return await kb_instance._get_lightrag_instance(db_id)

        except Exception as e:
            logger.error(f"Failed to get LightRAG instance for {db_id}: {e}")
            return None

    def is_lightrag_database(self, db_id: str) -> bool:
        """
        检查数据库是否是 LightRAG 类型

        Args:
            db_id: 数据库ID

        Returns:
            是否是 LightRAG 类型的数据库
        """
        if db_id not in self.global_databases_meta:
            return False

        kb_type = self.global_databases_meta[db_id].get("kb_type", "lightrag")
        return kb_type == "lightrag"

    def get_lightrag_databases(self) -> list[dict]:
        """
        获取所有 LightRAG 类型的数据库

        Returns:
            LightRAG 数据库列表
        """
        lightrag_databases = []

        all_databases = self.get_databases()["databases"]
        for db in all_databases:
            if db.get("kb_type", "lightrag") == "lightrag":
                lightrag_databases.append(db)

        return lightrag_databases
