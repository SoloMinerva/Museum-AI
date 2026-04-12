"""
按博物馆拆分数据并分别导入 LightRAG 知识图谱。

从 combined_museum_all_chunked.txt 按"收藏单位"拆分为：
  - hubei_chunked.txt (湖北省博物馆)
  - national_chunked.txt (中国国家博物馆)
然后分别创建两个 LightRAG 知识库并逐条导入。

用法（项目根目录）:
  museum/Scripts/python.exe test/split_and_import.py --split-only        # 只拆分不导入
  museum/Scripts/python.exe test/split_and_import.py --museum hubei      # 只导入湖北省博
  museum/Scripts/python.exe test/split_and_import.py --museum national   # 只导入国博
  museum/Scripts/python.exe test/split_and_import.py                     # 拆分 + 全部导入
  museum/Scripts/python.exe test/split_and_import.py --resume            # 断点续传
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "saves" / "knowledge_base_data" / "lightrag_data"
COMBINED_FILE = DATA_DIR / "combined_museum_all_chunked.txt"
HUBEI_FILE = DATA_DIR / "hubei_chunked.txt"
NATIONAL_FILE = DATA_DIR / "national_chunked.txt"

LOG_FILE = PROJECT_ROOT / "test" / "split_import.log"
PROGRESS_FILE = PROJECT_ROOT / "test" / "split_import_progress.json"

MAX_RETRIES = 3
CONCURRENCY = 2
SINGLE_ITEM_TIMEOUT = 300


MUSEUM_CONFIG = {
    "hubei": {
        "keyword": "湖北省博物馆",
        "data_file": HUBEI_FILE,
        "db_name": "湖北省博物馆知识图谱",
        "db_desc": "湖北省博物馆文物数据，LightRAG知识图谱",
    },
    "national": {
        "keyword": "中国国家博物馆",
        "data_file": NATIONAL_FILE,
        "db_name": "中国国家博物馆知识图谱",
        "db_desc": "中国国家博物馆文物数据，LightRAG知识图谱",
    },
}


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def split_data():
    """按博物馆拆分 combined_museum_all_chunked.txt"""
    if not COMBINED_FILE.is_file():
        log(f"错误：文件不存在: {COMBINED_FILE}")
        sys.exit(1)

    text = COMBINED_FILE.read_text(encoding="utf-8")
    chunks = [c.strip() for c in text.split("\n---\n") if c.strip()]
    log(f"总共 {len(chunks)} 条文物数据")

    hubei_chunks = []
    national_chunks = []

    for chunk in chunks:
        if "湖北省博物馆" in chunk:
            hubei_chunks.append(chunk)
        elif "中国国家博物馆" in chunk:
            national_chunks.append(chunk)
        else:
            log(f"警告：无法归类的文物: {chunk[:60]}")

    # 写入拆分后的文件
    HUBEI_FILE.write_text("\n---\n".join(hubei_chunks), encoding="utf-8")
    NATIONAL_FILE.write_text("\n---\n".join(national_chunks), encoding="utf-8")

    log(f"湖北省博: {len(hubei_chunks)} 条 → {HUBEI_FILE.name}")
    log(f"国博: {len(national_chunks)} 条 → {NATIONAL_FILE.name}")

    return len(hubei_chunks), len(national_chunks)


def load_chunks(filepath: Path) -> list[str]:
    text = filepath.read_text(encoding="utf-8")
    chunks = text.split("\n---\n")
    return [c.strip() for c in chunks if c.strip()]


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


async def check_neo4j_count(db_id: str) -> dict:
    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD", "museum123456")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    try:
        with driver.session(database="chunk-entity-relation") as session:
            nodes = session.run(f"MATCH (n:`{db_id}`) RETURN count(n) AS c").single()["c"]
            rels = session.run(f"MATCH (:`{db_id}`)-[r]->(:`{db_id}`) RETURN count(r) AS c").single()["c"]
        return {"nodes": nodes, "relations": rels}
    except Exception as e:
        return {"nodes": -1, "relations": -1, "error": str(e)}
    finally:
        driver.close()


async def import_single_chunk(knowledge_base, db_id, chunk_text, chunk_index, total):
    first_line = chunk_text.split("\n")[0].strip()
    artifact_name = first_line[:40] if first_line else f"chunk_{chunk_index}"

    tmp_dir = PROJECT_ROOT / "test" / "tmp_chunks"
    tmp_dir.mkdir(exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        tmp_file = tmp_dir / f"chunk_{chunk_index}.txt"
        tmp_file.write_text(chunk_text, encoding="utf-8")

        try:
            log(f"[{chunk_index}/{total}] 导入: {artifact_name} (尝试 {attempt}/{MAX_RETRIES})")
            t0 = time.time()

            results = await asyncio.wait_for(
                knowledge_base.add_content(db_id, [str(tmp_file.resolve())], {"content_type": "file"}),
                timeout=SINGLE_ITEM_TIMEOUT,
            )

            elapsed = time.time() - t0
            status = results[0].get("status", "unknown") if results else "no_result"

            if status == "done":
                log(f"[{chunk_index}/{total}] 成功! {artifact_name} ({elapsed:.1f}s)")
                return True, ""
            else:
                error = results[0].get("error", "") if results else ""
                log(f"[{chunk_index}/{total}] 失败! 状态: {status}, 错误: {error}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(5 * attempt)

        except asyncio.TimeoutError:
            log(f"[{chunk_index}/{total}] 超时! {artifact_name}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(10 * attempt)

        except Exception as e:
            log(f"[{chunk_index}/{total}] 异常! {artifact_name}: {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(5 * attempt)

        finally:
            if tmp_file.exists():
                tmp_file.unlink()

    return False, f"重试 {MAX_RETRIES} 次后仍失败: {artifact_name}"


async def import_museum(museum_key: str, resume: bool = False):
    """导入单个博物馆的数据"""
    cfg = MUSEUM_CONFIG[museum_key]
    data_file = cfg["data_file"]

    if not data_file.is_file():
        log(f"错误：数据文件不存在: {data_file}，请先运行 --split-only")
        return

    chunks = load_chunks(data_file)
    log(f"\n{'='*60}")
    log(f"=== 开始导入: {cfg['db_name']} ({len(chunks)} 条) ===")
    log(f"{'='*60}")

    # 加载进度
    all_progress = load_progress()
    progress = all_progress.get(museum_key, {"done": [], "failed": [], "db_id": ""})

    if resume and progress["done"]:
        log(f"断点续传: 已完成 {len(progress['done'])} 条，跳过")

    from src import config
    from src.knowledge import knowledge_base

    embed_info = config.embed_model_names.get("siliconflow/BAAI/bge-m3")
    if not embed_info:
        log("错误: 未找到 siliconflow/BAAI/bge-m3 embedding 配置")
        return

    # 创建或复用数据库
    db_id = progress.get("db_id", "")

    if not db_id:
        for did, meta in knowledge_base.global_databases_meta.items():
            if meta.get("name") == cfg["db_name"]:
                log(f"发现同名数据库: {did}，复用")
                db_id = did
                break

    if not db_id:
        log(f"创建新数据库: {cfg['db_name']}")
        info = await knowledge_base.create_database(
            cfg["db_name"],
            cfg["db_desc"],
            kb_type="lightrag",
            embed_info=embed_info,
        )
        db_id = info["db_id"]
        log(f"数据库已创建: {db_id}")

    progress["db_id"] = db_id
    log(f"db_id: {db_id}")

    neo4j_before = await check_neo4j_count(db_id)
    log(f"导入前 Neo4j: 节点={neo4j_before.get('nodes', '?')}, 关系={neo4j_before.get('relations', '?')}")

    total_start = time.time()
    success_count = len(progress["done"])
    fail_count = 0
    done_set = set(progress["done"])
    processed_count = 0  # 本轮已处理数

    semaphore = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()  # 保护 progress 写入

    async def _import_one(i, chunk):
        nonlocal success_count, fail_count, processed_count
        async with semaphore:
            ok, err = await import_single_chunk(knowledge_base, db_id, chunk, i, len(chunks))

        async with lock:
            if ok:
                success_count += 1
                progress["done"].append(i)
            else:
                fail_count += 1
                progress["failed"].append({"index": i, "error": err, "first_line": chunk.split("\n")[0][:60]})

            processed_count += 1
            if processed_count % 10 == 0:
                all_progress[museum_key] = progress
                save_progress(all_progress)
                neo4j_now = await check_neo4j_count(db_id)
                elapsed = time.time() - total_start
                rate = success_count / (elapsed / 60) if elapsed > 0 else 0
                remaining = len(chunks) - success_count - fail_count
                eta_min = remaining / rate if rate > 0 else 0
                log(f"--- 进度: {success_count}/{len(chunks)} | "
                    f"Neo4j: 节点={neo4j_now.get('nodes', '?')}, 关系={neo4j_now.get('relations', '?')} | "
                    f"ETA: {eta_min:.0f}min ---")

    tasks = []
    for i, chunk in enumerate(chunks, 1):
        if i in done_set:
            continue
        tasks.append(_import_one(i, chunk))

    log(f"并发数: {CONCURRENCY}, 待导入: {len(tasks)} 条")
    await asyncio.gather(*tasks)

    # 保存最终进度
    all_progress[museum_key] = progress
    save_progress(all_progress)

    total_elapsed = time.time() - total_start
    neo4j_final = await check_neo4j_count(db_id)

    log(f"\n{'='*60}")
    log(f"=== {cfg['db_name']} 导入完成 ===")
    log(f"成功: {success_count}/{len(chunks)}")
    log(f"失败: {fail_count}")
    log(f"耗时: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
    log(f"Neo4j: 节点={neo4j_final.get('nodes', '?')}, 关系={neo4j_final.get('relations', '?')}")
    log(f"db_id: {db_id}")
    log(f"{'='*60}")

    # 清理
    tmp_dir = PROJECT_ROOT / "test" / "tmp_chunks"
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return db_id


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-only", action="store_true", help="只拆分数据，不导入")
    parser.add_argument("--museum", choices=["hubei", "national"], help="只导入指定博物馆")
    parser.add_argument("--resume", action="store_true", help="断点续传")
    args = parser.parse_args()

    if not args.resume:
        LOG_FILE.write_text("", encoding="utf-8")

    # 第一步：拆分数据
    log("=== 拆分数据 ===")
    split_data()

    if args.split_only:
        log("拆分完成，退出")
        return

    # 第二步：导入
    if args.museum:
        await import_museum(args.museum, resume=args.resume)
    else:
        # 先导入湖北，再导入国博
        await import_museum("hubei", resume=args.resume)
        await import_museum("national", resume=args.resume)

    log("\n=== 全部完成 ===")


if __name__ == "__main__":
    asyncio.run(main())
