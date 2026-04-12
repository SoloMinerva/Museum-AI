"""
对比 ChromaDB vs Neo4j(LightRAG) 检索效果。
同样问题、同样 rerank，由易到难。

用法:
  museum\Scripts\python.exe test\test_chroma_vs_neo4j.py
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

CHROMA_DB_ID = "kb_d6f0936fffbeceb73dcd78a442dad8cb"
NEO4J_DB_ID = "kb_08e43845ea616e6dc3ebb630f18250bd"

# 由易到难的测试问题
TEST_QUESTIONS = [
    # --- 简单：直接检索单个文物 ---
    ("曾侯乙编钟是什么？", "naive"),
    ("越王勾践剑的尺寸是多少？", "naive"),
    ("四爱图梅瓶是哪个朝代的？", "naive"),
    # --- 中等：某类/某地文物 ---
    ("湖北省博物馆有哪些玉器？", "naive"),
    ("战国时期有哪些青铜器？", "naive"),
    ("九连墩墓出土了哪些文物？", "naive"),
    # --- 复杂：跨文物比较/综合分析 ---
    ("从商代到战国，青铜器的制作工艺有什么变化？", "mix"),
    ("曾侯乙墓出土的文物反映了怎样的礼乐文化？", "mix"),
    ("古代玉器在不同朝代的用途有什么变化？", "mix"),
    ("比较不同朝代瓷器的釉色和纹饰特点。", "mix"),
]


def show_top3(results: list[dict], label: str):
    sorted_r = sorted(results, key=lambda x: x.get("rerank_score", x.get("score", 0)), reverse=True)
    for j, r in enumerate(sorted_r[:3], 1):
        content = r.get("content", "")[:100].replace("\n", " ")
        rerank = r.get("rerank_score", None)
        raw = r.get("score", 0)
        score_str = f"rerank={rerank:.3f}" if rerank is not None else f"score={raw:.3f}"
        print(f"  {label} Top{j} [{score_str}]: {content}...")


async def main():
    from src import config
    from src.knowledge import knowledge_base

    print("=" * 78)
    print("ChromaDB vs Neo4j(LightRAG) 检索对比测试")
    print(f"Rerank: {config.enable_reranker} | Reranker: {config.reranker}")
    print(f"ChromaDB: {CHROMA_DB_ID}")
    print(f"Neo4j:    {NEO4J_DB_ID}")
    print("=" * 78)

    for i, (question, mode) in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'─' * 78}")
        difficulty = "简单" if mode == "naive" else "复杂"
        print(f"[Q{i}] [{difficulty}] {question}")
        print(f"{'─' * 78}")

        # --- ChromaDB ---
        t0 = time.time()
        try:
            chroma_results = await knowledge_base.aquery(question, CHROMA_DB_ID, top_k=10)
            ct = time.time() - t0
            print(f"  ChromaDB: {len(chroma_results)} 条 | {ct:.1f}s")
            show_top3(chroma_results, "Chroma")
        except Exception as e:
            ct = time.time() - t0
            print(f"  ChromaDB 失败: {e} | {ct:.1f}s")

        # --- Neo4j (LightRAG) ---
        t0 = time.time()
        try:
            neo4j_results = await knowledge_base.aquery(question, NEO4J_DB_ID, mode=mode)
            nt = time.time() - t0
            print(f"  Neo4j:    {len(neo4j_results)} 条 | {nt:.1f}s")
            show_top3(neo4j_results, "Neo4j")
        except Exception as e:
            nt = time.time() - t0
            print(f"  Neo4j 失败: {e} | {nt:.1f}s")

    print(f"\n{'=' * 78}")
    print("测试完成")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    asyncio.run(main())
