"""
将湖北省博物馆和中国国家博物馆数据分别导入到独立的 LightRAG 数据库中。

用法：
    python import_museums_to_lightrag.py

- Embedding: 本地 Ollama bge-m3
- LLM: 硅基流动免费 Qwen (Qwen/Qwen2.5-7B-Instruct)
- 并发数: 2
"""

import asyncio
import os
import sys

# 确保项目根目录在 sys.path 中
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# 加载 .env
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT_DIR, ".env"), override=False)

# 设置 LightRAG LLM 为硅基流动免费 Qwen (覆盖 .env 中的默认值)
os.environ["LIGHTRAG_LLM_PROVIDER"] = "siliconflow"
os.environ["LIGHTRAG_LLM_NAME"] = "Qwen/Qwen2.5-7B-Instruct"

from src.knowledge.implementations.lightrag import LightRagKB

WORK_DIR = r"C:\lightrag_work"

# 两个博物馆的配置
MUSEUMS = [
    {
        "name": "湖北省博物馆",
        "description": "湖北省博物馆文物知识库",
        "file": os.path.join(WORK_DIR, "hubei_chunked.txt"),
    },
    {
        "name": "中国国家博物馆",
        "description": "中国国家博物馆文物知识库",
        "file": os.path.join(WORK_DIR, "national_chunked.txt"),
    },
]

# 本地 Ollama embedding 配置
EMBED_INFO = {
    "name": "bge-m3",
    "base_url": "http://localhost:11434/v1/embeddings",
    "api_key": "OLLAMA_API_KEY",
    "dimension": 1024,
}

# 硅基流动免费 Qwen LLM 配置
LLM_INFO = {
    "provider": "siliconflow",
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "model_spec": "siliconflow/Qwen/Qwen2.5-7B-Instruct",
}

BATCH_SIZE = 50  # 每批插入的文物条数


def split_by_separator(text: str, separator: str = "---") -> list[str]:
    """按分隔符切分文本，返回非空的条目列表"""
    parts = text.split(separator)
    return [p.strip() for p in parts if p.strip()]


async def import_museum(kb: LightRagKB, museum_cfg: dict):
    """导入单个博物馆的数据"""
    name = museum_cfg["name"]
    filepath = museum_cfg["file"]

    print(f"\n{'='*60}")
    print(f"开始处理: {name}")
    print(f"数据文件: {filepath}")

    # 1. 创建数据库
    db_info = kb.create_database(
        database_name=name,
        description=museum_cfg["description"],
        embed_info=EMBED_INFO,
        llm_info=LLM_INFO,
    )
    db_id = db_info["db_id"]
    print(f"已创建数据库: {db_id}")

    # 2. 获取 LightRAG 实例
    rag = await kb._get_lightrag_instance(db_id)
    if not rag:
        print(f"[ERROR] 无法创建 LightRAG 实例: {db_id}")
        return

    # 修改并发数为 2
    rag.llm_model_max_async = 2
    rag.embedding_func_max_async = 2

    # 3. 读取数据并分条
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    items = split_by_separator(text, "---")
    total = len(items)
    print(f"共 {total} 条文物，分 {(total + BATCH_SIZE - 1) // BATCH_SIZE} 批导入 (每批 {BATCH_SIZE} 条)")

    # 4. 分批插入
    for i in range(0, total, BATCH_SIZE):
        batch = items[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        batch_text = "\n---\n".join(batch)

        print(f"  [{name}] 批次 {batch_num}: 第 {i+1}-{min(i+BATCH_SIZE, total)} 条 ...", end=" ", flush=True)

        try:
            await rag.ainsert(
                input=batch_text,
                split_by_character="---",
                split_by_character_only=True,
            )
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"[{name}] 导入完成! db_id={db_id}")
    return db_id


async def main():
    print("="*60)
    print("博物馆数据导入 LightRAG")
    print(f"  LLM: siliconflow / Qwen/Qwen2.5-7B-Instruct (免费)")
    print(f"  Embedding: ollama / bge-m3 (本地)")
    print(f"  并发数: 2")
    print("="*60)

    # 检查数据文件是否存在
    for m in MUSEUMS:
        if not os.path.exists(m["file"]):
            print(f"[ERROR] 数据文件不存在: {m['file']}")
            return

    kb = LightRagKB(work_dir=WORK_DIR)

    db_ids = []
    for museum_cfg in MUSEUMS:
        db_id = await import_museum(kb, museum_cfg)
        if db_id:
            db_ids.append((museum_cfg["name"], db_id))

    print(f"\n{'='*60}")
    print("全部导入完成!")
    for name, db_id in db_ids:
        print(f"  {name}: {db_id}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
