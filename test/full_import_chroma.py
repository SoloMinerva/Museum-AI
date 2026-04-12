"""
全量导入 1359 条文物到 ChromaDB 向量库。

ChromaDB 只做 embedding 向量化，不调用 LLM，速度很快。
使用本地 Ollama bge-m3 embedding。

用法:
  museum\\Scripts\\python.exe test\\full_import_chroma.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

INPUT_FILE = PROJECT_ROOT / "saves" / "knowledge_base_data" / "lightrag_data" / "combined_museum_all_chunked.txt"
LOG_FILE = PROJECT_ROOT / "test" / "full_import_chroma.log"


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


async def main():
    LOG_FILE.write_text("", encoding="utf-8")

    log("=" * 60)
    log("=== 全量 ChromaDB 导入开始 ===")
    log("=" * 60)

    if not INPUT_FILE.is_file():
        log(f"错误：文件不存在: {INPUT_FILE}")
        sys.exit(1)

    log(f"数据文件: {INPUT_FILE.name} ({INPUT_FILE.stat().st_size:,} bytes)")

    from src import config
    from src.knowledge import knowledge_base

    # Embedding: 本地 Ollama bge-m3
    embed_info = config.embed_model_names.get("ollama/bge-m3")
    if not embed_info:
        log("错误: 未找到 ollama/bge-m3 embedding 配置")
        sys.exit(1)

    log(f"Embedding: ollama/bge-m3 (本地)")

    # 创建或复用数据库
    db_name = "全馆文物向量库-1359"
    db_id = ""

    for did, meta in knowledge_base.global_databases_meta.items():
        if meta.get("name") == db_name:
            log(f"发现同名数据库: {did}，复用")
            db_id = did
            break

    if not db_id:
        log(f"创建新数据库: {db_name}")
        info = await knowledge_base.create_database(
            db_name,
            "全馆1359条文物，ChromaDB向量存储，本地Ollama bge-m3 embedding",
            kb_type="chroma",
            embed_info=embed_info,
        )
        db_id = info["db_id"]
        log(f"数据库已创建: {db_id}")

    log(f"db_id: {db_id}")

    # 开始导入
    log(f"开始导入... (按 --- 分隔切块)")
    log("ChromaDB 只做 embedding，不调用 LLM，速度很快")
    t0 = time.time()

    results = await knowledge_base.add_content(
        db_id,
        [str(INPUT_FILE.resolve())],
        {
            "content_type": "file",
            "use_delimiter_split": True,
            "delimiter": "---",
        },
    )

    elapsed = time.time() - t0

    for r in results:
        status = r.get("status", "unknown")
        filename = r.get("filename", "")
        log(f"文件: {filename} | 状态: {status}")
        if r.get("error"):
            log(f"错误: {r['error']}")

    log(f"导入耗时: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    log(f"db_id: {db_id}")

    # 检查文档数
    try:
        kb_instance = knowledge_base._get_kb_for_database(db_id)
        collection = await kb_instance._get_chroma_collection(db_id)
        if collection:
            count = collection.count()
            log(f"ChromaDB 文档数: {count}")
    except Exception as e:
        log(f"检查文档数失败: {e}")

    log("=" * 60)
    log("=== 导入完成 ===")
    log("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
