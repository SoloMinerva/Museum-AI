"""
全量导入 1359 条文物到 LightRAG → Neo4j 知识图谱。

特点：
- 逐条文物导入，每条独立写入临时文件后调用 add_content
- 本地 Ollama bge-m3 embedding
- API 超时自动重试（最多 3 次）
- 详细进度日志 + 失败记录，支持断点续传

用法（项目根目录）:
  museum\\Scripts\\python.exe test\\full_import_neo4j.py
  museum\\Scripts\\python.exe test\\full_import_neo4j.py --resume          # 断点续传，跳过已成功的
  museum\\Scripts\\python.exe test\\full_import_neo4j.py --use-existing <db_id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

INPUT_FILE = PROJECT_ROOT / "saves" / "knowledge_base_data" / "lightrag_data" / "combined_museum_all_chunked.txt"
LOG_FILE = PROJECT_ROOT / "test" / "full_import_neo4j.log"
PROGRESS_FILE = PROJECT_ROOT / "test" / "full_import_neo4j_progress.json"

MAX_RETRIES = 3
# 单条文物导入超时（秒）：LLM 抽取实体 + embedding + Neo4j 写入
SINGLE_ITEM_TIMEOUT = 300


def log(msg: str):
    """同时输出到终端和日志文件"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_chunks(filepath: Path) -> list[str]:
    """按 --- 分隔符切分文物数据，返回非空 chunk 列表"""
    text = filepath.read_text(encoding="utf-8")
    chunks = text.split("\n---\n")
    # 过滤空白 chunk
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks


def load_progress() -> dict:
    """加载进度文件"""
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"done": [], "failed": []}
    return {"done": [], "failed": []}


def save_progress(progress: dict):
    """保存进度文件"""
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


async def check_neo4j_count(db_id: str) -> dict:
    """查询 Neo4j 中该 workspace 的节点和关系数"""
    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD", "museum123456")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    try:
        with driver.session(database="chunk-entity-relation") as session:
            nodes = session.run(
                f"MATCH (n:`{db_id}`) RETURN count(n) AS c"
            ).single()["c"]
            rels = session.run(
                f"MATCH (:`{db_id}`)-[r]->(:`{db_id}`) RETURN count(r) AS c"
            ).single()["c"]
        return {"nodes": nodes, "relations": rels}
    except Exception as e:
        return {"nodes": -1, "relations": -1, "error": str(e)}
    finally:
        driver.close()


async def import_single_chunk(
    knowledge_base, db_id: str, chunk_text: str, chunk_index: int, total: int
) -> tuple[bool, str]:
    """
    导入单条文物，带超时和重试。
    返回 (success, error_msg)
    """
    # 提取文物名称用于日志
    first_line = chunk_text.split("\n")[0].strip()
    artifact_name = first_line[:40] if first_line else f"chunk_{chunk_index}"

    for attempt in range(1, MAX_RETRIES + 1):
        # 写入临时文件
        tmp_dir = PROJECT_ROOT / "test" / "tmp_chunks"
        tmp_dir.mkdir(exist_ok=True)
        tmp_file = tmp_dir / f"chunk_{chunk_index}.txt"
        tmp_file.write_text(chunk_text, encoding="utf-8")

        try:
            log(f"[{chunk_index}/{total}] 导入: {artifact_name} (尝试 {attempt}/{MAX_RETRIES})")
            t0 = time.time()

            results = await asyncio.wait_for(
                knowledge_base.add_content(
                    db_id,
                    [str(tmp_file.resolve())],
                    {"content_type": "file"},
                ),
                timeout=SINGLE_ITEM_TIMEOUT,
            )

            elapsed = time.time() - t0
            status = results[0].get("status", "unknown") if results else "no_result"
            error = results[0].get("error", "") if results else ""

            if status == "done":
                log(f"[{chunk_index}/{total}] 成功! {artifact_name} ({elapsed:.1f}s)")
                return True, ""
            else:
                log(f"[{chunk_index}/{total}] 失败! 状态: {status}, 错误: {error}")
                if attempt < MAX_RETRIES:
                    wait = 5 * attempt
                    log(f"  等待 {wait}s 后重试...")
                    await asyncio.sleep(wait)

        except asyncio.TimeoutError:
            elapsed = time.time() - t0
            log(f"[{chunk_index}/{total}] 超时! {artifact_name} ({elapsed:.1f}s > {SINGLE_ITEM_TIMEOUT}s)")
            if attempt < MAX_RETRIES:
                wait = 10 * attempt
                log(f"  等待 {wait}s 后重试...")
                await asyncio.sleep(wait)

        except Exception as e:
            elapsed = time.time() - t0
            log(f"[{chunk_index}/{total}] 异常! {artifact_name}: {e} ({elapsed:.1f}s)")
            if attempt < MAX_RETRIES:
                wait = 5 * attempt
                log(f"  等待 {wait}s 后重试...")
                await asyncio.sleep(wait)

        finally:
            # 清理临时文件
            if tmp_file.exists():
                tmp_file.unlink()

    return False, f"重试 {MAX_RETRIES} 次后仍失败: {artifact_name}"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="断点续传，跳过已成功的 chunk")
    parser.add_argument("--use-existing", type=str, default="", help="使用已有 db_id")
    args = parser.parse_args()

    if not args.resume:
        LOG_FILE.write_text("", encoding="utf-8")

    log("=" * 60)
    log("=== 全量 Neo4j 导入开始 ===")
    log("=" * 60)

    if not INPUT_FILE.is_file():
        log(f"错误：文件不存在: {INPUT_FILE}")
        sys.exit(1)

    # 切分文物
    chunks = load_chunks(INPUT_FILE)
    log(f"数据文件: {INPUT_FILE.name} ({INPUT_FILE.stat().st_size:,} bytes)")
    log(f"文物总数: {len(chunks)} 条")

    # 加载进度
    progress = load_progress() if args.resume else {"done": [], "failed": []}

    if args.resume and progress["done"]:
        log(f"断点续传: 已完成 {len(progress['done'])} 条，跳过")

    from src import config
    from src.knowledge import knowledge_base

    # Embedding: 本地 Ollama bge-m3
    embed_info = config.embed_model_names.get("ollama/bge-m3")
    if not embed_info:
        log("错误: 未找到 ollama/bge-m3 embedding 配置")
        sys.exit(1)

    log(f"Embedding: ollama/bge-m3 (本地)")
    log(f"LLM: {os.environ.get('LIGHTRAG_LLM_PROVIDER', '?')}/{os.environ.get('LIGHTRAG_LLM_NAME', '?')}")
    log(f"并发: 2, 单条超时: {SINGLE_ITEM_TIMEOUT}s, 最大重试: {MAX_RETRIES}")

    # 创建或复用数据库
    db_id = args.use_existing.strip()
    db_name = "全馆文物知识图谱-1359"

    if not db_id:
        for did, meta in knowledge_base.global_databases_meta.items():
            if meta.get("name") == db_name:
                log(f"发现同名数据库: {did}，复用")
                db_id = did
                break

    if not db_id:
        log(f"创建新数据库: {db_name}")
        info = await knowledge_base.create_database(
            db_name,
            "全馆1359条文物，LightRAG知识图谱，本地Ollama embedding",
            kb_type="lightrag",
            embed_info=embed_info,
        )
        db_id = info["db_id"]
        log(f"数据库已创建: {db_id}")

    log(f"db_id: {db_id}")

    # 导入前检查 Neo4j
    neo4j_before = await check_neo4j_count(db_id)
    log(f"导入前 Neo4j: 节点={neo4j_before.get('nodes', '?')}, 关系={neo4j_before.get('relations', '?')}")

    total_start = time.time()
    success_count = len(progress["done"])
    fail_count = 0
    done_set = set(progress["done"])

    for i, chunk in enumerate(chunks, 1):
        # 断点续传：跳过已完成的
        if i in done_set:
            continue

        ok, err = await import_single_chunk(knowledge_base, db_id, chunk, i, len(chunks))

        if ok:
            success_count += 1
            progress["done"].append(i)
        else:
            fail_count += 1
            progress["failed"].append({"index": i, "error": err, "first_line": chunk.split("\n")[0][:60]})
            log(f"  记录失败: {err}")

        # 每 10 条保存一次进度 + 检查 Neo4j
        if i % 10 == 0:
            save_progress(progress)
            neo4j_now = await check_neo4j_count(db_id)
            elapsed = time.time() - total_start
            rate = success_count / (elapsed / 60) if elapsed > 0 else 0
            remaining = len(chunks) - success_count - fail_count
            eta_min = remaining / rate if rate > 0 else 0
            log(f"--- 进度: {success_count}/{len(chunks)} 成功, {fail_count} 失败 | "
                f"Neo4j: 节点={neo4j_now.get('nodes', '?')}, 关系={neo4j_now.get('relations', '?')} | "
                f"速率: {rate:.1f}条/min, 预计剩余: {eta_min:.0f}min ---")

    # 最终保存进度
    save_progress(progress)

    total_elapsed = time.time() - total_start
    neo4j_final = await check_neo4j_count(db_id)

    log("")
    log("=" * 60)
    log("=== 导入完成 ===")
    log(f"成功: {success_count}/{len(chunks)}")
    log(f"失败: {fail_count}")
    log(f"总耗时: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
    log(f"Neo4j 最终: 节点={neo4j_final.get('nodes', '?')}, 关系={neo4j_final.get('relations', '?')}")
    log(f"db_id: {db_id}")

    if progress["failed"]:
        log(f"\n失败列表 ({len(progress['failed'])} 条):")
        for item in progress["failed"]:
            log(f"  [{item['index']}] {item['first_line']} - {item['error']}")
        log(f"\n可使用 --resume 重新运行以重试失败项")

    log("=" * 60)

    # 清理临时目录
    tmp_dir = PROJECT_ROOT / "test" / "tmp_chunks"
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
