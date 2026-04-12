"""
批量导入图片 embedding 到 ChromaDB

从本地已下载的图片生成 CN-CLIP embedding，存入 ChromaDB 的图片 collection。

用法：
    python scripts/import_image_embeddings.py
    python scripts/import_image_embeddings.py --db-id kb_xxx  # 指定目标数据库
    python scripts/import_image_embeddings.py --batch-size 32  # 调整批次大小
"""

import json
import os
import sys
import time
import argparse
import traceback
from pathlib import Path

import torch
import chromadb
from chromadb.config import Settings
from PIL import Image

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cn_clip.clip.utils import create_model, image_transform
import cn_clip.clip as clip

# ============ 配置 ============
CHROMA_DB_PATH = PROJECT_ROOT / "saves" / "knowledge_base_data" / "chroma_data" / "chromadb"
MAPPING_FILE = PROJECT_ROOT / "data" / "image_mapping.json"
IMAGE_DIR = PROJECT_ROOT / "data" / "museum_images"
MODEL_PATH = PROJECT_ROOT / "saves" / "model" / "clip_finetune" / "checkpoints" / "best.pt"

# 默认挂载的 db_id（全馆文物向量库-1359）
DEFAULT_DB_ID = "kb_d6f0936fffbeceb73dcd78a442dad8cb"


def load_model():
    """加载 CN-CLIP 模型"""
    print("[1/4] 加载 CN-CLIP 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}")

    pretrained = torch.load(str(MODEL_PATH), map_location="cpu")
    model = create_model("ViT-B-16@RoBERTa-wwm-ext-base-chinese", checkpoint=pretrained)
    model = model.to(device)
    model.eval()

    preprocess = image_transform()
    print("  模型加载完成")
    return model, preprocess, device


def load_mapping():
    """加载图片映射数据"""
    print("[2/4] 加载映射数据...")

    # 优先用 mapping 文件
    if MAPPING_FILE.exists():
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        print(f"  从 mapping 文件加载: {len(mapping)} 条")
        return mapping

    # 没有 mapping 文件，从原始 JSON + 本地图片目录重建
    print("  mapping 文件不存在，从原始数据重建...")
    all_items = []
    for src_file in ["hbkgy_all_kb_enriched.json", "chnmuseum_all_kb_enriched.json"]:
        fpath = PROJECT_ROOT / src_file
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_items.extend(data)

    downloaded_files = {f.split("_", 1)[0]: f for f in os.listdir(IMAGE_DIR)} if IMAGE_DIR.exists() else {}

    mapping = []
    for i, item in enumerate(all_items):
        idx_str = f"{i:04d}"
        if idx_str in downloaded_files:
            mapping.append({
                "index": i,
                "name": item["name"],
                "museum": item["museum"],
                "description": item.get("description", ""),
                "image_url": item.get("image_url", ""),
                "detail_url": item.get("detail_url", ""),
                "local_path": f"data/museum_images/{downloaded_files[idx_str]}",
            })

    print(f"  重建完成: {len(mapping)} 条")
    return mapping


def generate_embedding(model, preprocess, device, image_path):
    """为单张图片生成 embedding"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features /= features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().astype("float32").flatten().tolist()


def get_or_create_image_collection(db_id):
    """获取或创建图片专用 ChromaDB collection"""
    print("[3/4] 连接 ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )

    collection_name = f"{db_id}_images"

    # 自定义 embedding function（占位，实际 embedding 由 CN-CLIP 外部生成）
    class DummyEmbeddingFunction:
        def __call__(self, input):
            return [[0.0] * 512 for _ in input]

    try:
        collection = client.get_collection(name=collection_name)
        existing_count = collection.count()
        print(f"  已有 collection: {collection_name}, 当前 {existing_count} 条")
    except Exception:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=DummyEmbeddingFunction(),
            metadata={
                "db_id": db_id,
                "embedding_model": "cn_clip_vit_b_16",
                "embedding_dimension": 512,
            },
        )
        print(f"  创建新 collection: {collection_name}")

    return collection


def main():
    parser = argparse.ArgumentParser(description="批量导入图片 embedding 到 ChromaDB")
    parser.add_argument("--db-id", default=DEFAULT_DB_ID, help="目标数据库 ID")
    parser.add_argument("--batch-size", type=int, default=32, help="每批写入条数")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="跳过已存在的")
    args = parser.parse_args()

    # 1. 加载模型
    model, preprocess, device = load_model()

    # 2. 加载映射数据
    mapping = load_mapping()
    if not mapping:
        print("没有可导入的数据")
        return

    # 3. 连接 ChromaDB
    collection = get_or_create_image_collection(args.db_id)

    # 检查已有数据，跳过已导入的
    existing_ids = set()
    if args.skip_existing and collection.count() > 0:
        # 分批获取所有已有 ID
        all_existing = collection.get(include=[])
        existing_ids = set(all_existing["ids"])
        print(f"  已有 {len(existing_ids)} 条，将跳过")

    # 过滤掉已存在的
    to_import = []
    for item in mapping:
        doc_id = f"img_{item['index']:04d}"
        if doc_id not in existing_ids:
            to_import.append(item)

    if not to_import:
        print("所有数据已导入，无需重复操作")
        return

    print(f"\n[4/4] 开始生成 embedding 并入库...")
    print(f"  待导入: {len(to_import)} 条")
    print(f"  批次大小: {args.batch_size}")
    print()

    # 4. 批量处理
    batch_ids = []
    batch_embeddings = []
    batch_documents = []
    batch_metadatas = []

    success = 0
    fail = 0
    start_time = time.time()

    for i, item in enumerate(to_import):
        idx = item["index"]
        local_path = PROJECT_ROOT / item["local_path"]

        if not local_path.exists():
            fail += 1
            continue

        try:
            # 生成 embedding
            embedding = generate_embedding(model, preprocess, device, str(local_path))

            doc_id = f"img_{idx:04d}"
            document = f"文物名称：{item['name']}\n收藏单位：{item['museum']}\n{item.get('description', '')}"
            metadata = {
                "name": item["name"],
                "museum": item["museum"],
                "image_url": item.get("image_url", ""),
                "detail_url": item.get("detail_url", ""),
                "local_path": item["local_path"],
                "index": idx,
                "source": "image_embedding",
            }

            batch_ids.append(doc_id)
            batch_embeddings.append(embedding)
            batch_documents.append(document)
            batch_metadatas.append(metadata)
            success += 1

        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  [FAIL] {idx:04d} {item['name']}: {str(e)[:80]}")

        # 写入一批
        if len(batch_ids) >= args.batch_size:
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
            )
            elapsed = time.time() - start_time
            speed = success / elapsed if elapsed > 0 else 0
            print(f"  [{success + fail}/{len(to_import)}] 已入库 {success} 条, 失败 {fail} 条, 速度 {speed:.1f} 条/秒")
            batch_ids.clear()
            batch_embeddings.clear()
            batch_documents.clear()
            batch_metadatas.clear()

    # 写入最后一批
    if batch_ids:
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )

    elapsed = time.time() - start_time
    total_in_collection = collection.count()

    print()
    print("=" * 50)
    print("导入完成!")
    print(f"  成功: {success}")
    print(f"  失败: {fail}")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  Collection 总数: {total_in_collection}")
    print(f"  Collection 名称: {args.db_id}_images")
    print("=" * 50)


if __name__ == "__main__":
    main()
