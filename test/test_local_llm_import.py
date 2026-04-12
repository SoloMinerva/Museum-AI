"""
测试本地 LLM (qwen3:8b) + 远程 embedding (siliconflow bge-m3) 导入效果。
只导入 3 条，观察实体抽取质量。

用法: museum/Scripts/python.exe test/test_local_llm_import.py
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


async def main():
    from src import config
    from src.knowledge import knowledge_base

    # 使用硅基流动 bge-m3 做 embedding
    embed_info = config.embed_model_names.get("siliconflow/BAAI/bge-m3")
    if not embed_info:
        print("错误: 未找到 siliconflow/BAAI/bge-m3 配置")
        return

    print(f"LLM: {os.getenv('LIGHTRAG_LLM_PROVIDER')}/{os.getenv('LIGHTRAG_LLM_NAME')}")
    print(f"Embedding: siliconflow/BAAI/bge-m3")

    # 创建测试数据库
    db_name = "测试-本地LLM导入"
    info = await knowledge_base.create_database(
        db_name,
        "测试本地qwen3:8b + 远程embedding效果",
        kb_type="lightrag",
        embed_info=embed_info,
    )
    db_id = info["db_id"]
    print(f"db_id: {db_id}")

    # 准备 3 条测试数据（从湖北省博中挑选有代表性的）
    test_chunks = [
        """【越王勾践剑】
收藏单位：湖北省博物馆
类别：青铜器

春秋晚期，长55.7cm，1965年江陵望山1号墓出土。
剑身满饰黑色菱形花纹，剑格正面镶蓝色琉璃，背面镶绿松石。剑首向外翻卷作圆箍形，内铸11道同心圆。
近格处有两行鸟篆铭文："越王鸠浅自作用剑"。此剑铸造精良，历经两千多年仍锋利无比，是青铜武器中的珍品。""",

        """【曾侯乙编钟】
收藏单位：湖北省博物馆
类别：青铜器

战国早期（约公元前433年），1978年湖北随州擂鼓墩曾侯乙墓出土。
全套编钟共65件，分三层八组悬挂在钟架上，总重量达2567公斤。
最大钟通高152.3厘米，重203.6公斤。每件钟都能奏出呈三度音阶的双音，整套编钟的音域跨五个半八度。
编钟是中国迄今发现数量最多、保存最好、音律最全、气势最宏伟的一套编钟。""",

        """【铜鼎】
收藏单位：湖北省博物馆
类别：青铜器

战国，高27.2，口径41cm，1986年荆门包山2号墓出土。"""
    ]

    # 逐条导入
    tmp_dir = PROJECT_ROOT / "test" / "tmp_test"
    tmp_dir.mkdir(exist_ok=True)

    for i, chunk in enumerate(test_chunks, 1):
        tmp_file = tmp_dir / f"test_{i}.txt"
        tmp_file.write_text(chunk, encoding="utf-8")

        print(f"\n--- 导入第 {i}/3 条 ---")
        print(f"内容: {chunk[:50]}...")
        t0 = time.time()

        try:
            results = await asyncio.wait_for(
                knowledge_base.add_content(db_id, [str(tmp_file.resolve())], {"content_type": "file"}),
                timeout=300,
            )
            elapsed = time.time() - t0
            status = results[0].get("status", "unknown") if results else "no_result"
            print(f"状态: {status}, 耗时: {elapsed:.1f}s")
        except Exception as e:
            print(f"错误: {e}")
        finally:
            if tmp_file.exists():
                tmp_file.unlink()

    # 等待后台处理完成（LightRAG 异步队列）
    print("\n等待 LightRAG 后台处理...")
    from neo4j import GraphDatabase
    d = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "museum123456"))

    for wait in range(60):  # 最多等 5 分钟
        await asyncio.sleep(5)
        with d.session(database="chunk-entity-relation") as s:
            nodes = s.run(f"MATCH (n:`{db_id}`) RETURN count(n) AS c").single()["c"]
            rels = s.run(f"MATCH (:`{db_id}`)-[r]->(:`{db_id}`) RETURN count(r) AS c").single()["c"]
        print(f"  [{wait*5}s] Neo4j: nodes={nodes}, rels={rels}")

        # 检查 LightRAG 日志看是否处理完
        import glob
        log_files = sorted(glob.glob("saves/knowledge_base_data/lightrag_data/logs/lightrag/lightrag_*.log"))
        if log_files:
            with open(log_files[-1], encoding="utf-8") as f:
                lines = f.readlines()
            # 找最新的 "Completed processing" 行
            completed = [l for l in lines if "Completed processing" in l and db_id[:8] in l.lower()]
            # 检查是否还在处理
            extracting = [l for l in lines[-5:] if "Extracting stage" in l or "Processing d-id" in l]
            if not extracting and nodes > 0:
                print("  后台处理似乎已完成")
                break

    # 查看抽取结果
    print(f"\n=== 最终结果 ===")
    with d.session(database="chunk-entity-relation") as s:
        nodes_count = s.run(f"MATCH (n:`{db_id}`) RETURN count(n) AS c").single()["c"]
        rels_count = s.run(f"MATCH (:`{db_id}`)-[r]->(:`{db_id}`) RETURN count(r) AS c").single()["c"]

        print(f"节点数: {nodes_count}")
        print(f"关系数: {rels_count}")

        # 列出所有节点
        print(f"\n--- 所有节点 ---")
        result = s.run(f"MATCH (n:`{db_id}`) RETURN n.entity_id AS name, labels(n) AS labels LIMIT 50")
        for r in result:
            print(f"  {r['name']}")

        # 列出所有关系
        print(f"\n--- 所有关系 ---")
        result = s.run(f"MATCH (a:`{db_id}`)-[r]->(b:`{db_id}`) RETURN a.entity_id AS from_node, type(r) AS rel, b.entity_id AS to_node LIMIT 50")
        for r in result:
            print(f"  {r['from_node']} --[{r['rel']}]--> {r['to_node']}")

    d.close()

    # 检查格式错误
    if log_files:
        with open(log_files[-1], encoding="utf-8") as f:
            lines = f.readlines()
        errors = [l.strip() for l in lines if "format error" in l and "00:2" in l or "format error" in l and "00:3" in l]
        print(f"\n--- 格式错误: {len(errors)} 条 ---")
        for e in errors[:10]:
            print(f"  {e[-120:]}")

    # 清理
    import shutil
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n测试完成。db_id: {db_id}")
    print("如需删除测试数据库，手动处理即可。")


if __name__ == "__main__":
    asyncio.run(main())
