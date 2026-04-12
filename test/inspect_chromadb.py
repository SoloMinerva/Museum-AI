"""查看本地 Chroma 持久化目录中的集合与样本数据。

用法（项目根目录）:
  museum\\Scripts\\python.exe test\\inspect_chromadb.py
  museum\\Scripts\\python.exe test\\inspect_chromadb.py --path saves/test_chromadb
"""

from __future__ import annotations

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import chromadb
from chromadb.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ChromaDB persistent data")
    parser.add_argument(
        "--path",
        default=os.path.join(project_root, "saves", "test_chromadb"),
        help="Chroma 持久化目录",
    )
    parser.add_argument("--limit", type=int, default=5, help="每个集合展示的条数")
    args = parser.parse_args()

    db_path = os.path.abspath(args.path)
    if not os.path.isdir(db_path):
        print(f"目录不存在: {db_path}")
        sys.exit(1)

    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False),
    )
    cols = client.list_collections()

    print("Chroma 路径:", db_path)
    print("集合数量:", len(cols))
    print()

    for col in cols:
        n = col.count()
        print("=" * 60)
        print(f"集合名: {col.name}")
        print(f"条数: {n}")
        meta = col.metadata or {}
        if meta:
            print("元数据:", meta)

        if n == 0:
            print()
            continue

        take = min(args.limit, n)
        data = col.get(limit=take, include=["documents", "metadatas"])
        ids = data.get("ids") or []
        print(f"\n前 {take} 条样本:")
        for i in range(len(ids)):
            print("-" * 40)
            print("id:", ids[i])
            m = data["metadatas"][i] or {}
            for k in ("name", "image_url", "detail_url"):
                if k in m and m[k]:
                    v = m[k]
                    if k != "name" and len(str(v)) > 70:
                        v = str(v)[:67] + "..."
                    print(f"  {k}: {v}")
            doc = data["documents"][i] or ""
            preview = doc.replace("\n", " ")[:240]
            more = "..." if len(doc) > 240 else ""
            print(f"  document: {preview}{more}")
        print()


if __name__ == "__main__":
    main()
