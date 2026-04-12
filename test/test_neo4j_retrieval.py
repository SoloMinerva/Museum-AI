"""
测试 1359 条全量知识图谱的检索效果。
测试路由分类 + LightRAG 检索 + Rerank。

用法:
  museum\\Scripts\\python.exe test\\test_neo4j_retrieval.py
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

DB_ID = "kb_08e43845ea616e6dc3ebb630f18250bd"

TEST_QUESTIONS = [
    # (问题, 期望路由, LightRAG mode)
    ("曾侯乙编钟是什么？", "simple", "naive"),
    ("越王勾践剑的尺寸是多少？", "simple", "naive"),
    ("湖北省博物馆有哪些玉器？", "simple", "naive"),
    ("战国时期有哪些青铜器？", "simple", "naive"),
    ("九连墩墓出土了哪些文物？", "simple", "naive"),
    ("从商代到战国，青铜器的制作工艺有什么变化？", "complex", "mix"),
    ("曾侯乙墓出土的文物反映了怎样的礼乐文化？", "complex", "mix"),
    ("两个博物馆各有什么特色藏品？", "complex", "mix"),
    ("古代玉器在不同朝代的用途有什么变化？", "complex", "mix"),
    ("比较不同朝代瓷器的釉色和纹饰特点。", "complex", "mix"),
]


async def main():
    from src import config
    from src.knowledge import knowledge_base

    # 直接导入 router 函数，避免触发 agents 包的完整导入链
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "router", PROJECT_ROOT / "src" / "agents" / "chatbot" / "router.py"
    )
    router_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(router_mod)
    classify_query = router_mod.classify_query
    get_lightrag_mode = router_mod.get_lightrag_mode

    print("=" * 70)
    print(f"知识图谱检索测试 | db_id: {DB_ID}")
    print(f"Rerank: {config.enable_reranker} | Reranker: {config.reranker}")
    print("=" * 70)

    for i, (question, expected_route, mode) in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'─' * 70}")
        print(f"[Q{i}] {question}")
        print(f"{'─' * 70}")
        print(f"mode: {mode} (预设)")

        # LightRAG 检索 + Rerank
        t0 = time.time()
        try:
            results = await knowledge_base.aquery(question, DB_ID, mode=mode)
            query_time = time.time() - t0
            print(f"检索: {len(results)} 条结果 | {query_time:.1f}s")

            # 显示 Top3 结果
            sorted_results = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
            for j, r in enumerate(sorted_results[:3], 1):
                content = r.get("content", "")[:120].replace("\n", " ")
                score = r.get("rerank_score", 0)
                print(f"  Top{j} [rerank={score:.3f}]: {content}...")

        except Exception as e:
            query_time = time.time() - t0
            print(f"检索失败: {e} | {query_time:.1f}s")

    print(f"\n{'=' * 70}")
    print("测试完成")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
