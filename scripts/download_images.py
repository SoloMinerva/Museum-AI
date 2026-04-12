"""
博物馆藏品图片批量下载脚本

从 enriched JSON 文件读取藏品数据，批量下载图片到本地，
生成 mapping.json 记录图片与藏品的一一对应关系。

用法：
    python scripts/download_images.py
    python scripts/download_images.py --retry   # 重试之前失败的
"""

import json
import os
import re
import time
import argparse
import hashlib
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============ 配置 ============
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "museum_images"
MAPPING_FILE = PROJECT_ROOT / "data" / "image_mapping.json"
FAILED_FILE = PROJECT_ROOT / "data" / "image_download_failed.json"

SOURCE_FILES = [
    PROJECT_ROOT / "hbkgy_all_kb_enriched.json",
    PROJECT_ROOT / "chnmuseum_all_kb_enriched.json",
]

MAX_WORKERS = 4        # 并发下载线程数（对博物馆CDN别太激进）
TIMEOUT = 15           # 单张图片下载超时（秒）
MAX_RETRIES = 3        # 单张图片最大重试次数
RETRY_DELAY = 2        # 重试间隔（秒）


def sanitize_filename(name: str, max_len: int = 50) -> str:
    """清理文件名，去除不合法字符"""
    name = re.sub(r'[\\/:*?"<>|\s]+', '_', name)
    name = name.strip('_.')
    return name[:max_len] if len(name) > max_len else name


def get_extension(url: str) -> str:
    """从URL提取文件扩展名"""
    path = url.split('?')[0]
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'):
        return ext
    return '.jpg'  # 默认


def download_one(item: dict) -> dict:
    """
    下载单张图片，返回结果记录。

    item 包含: index, name, museum, image_url, description, detail_url
    """
    index = item["index"]
    url = item["image_url"]
    museum_short = "hb" if "湖北" in item["museum"] else "guo"
    safe_name = sanitize_filename(item["name"])
    ext = get_extension(url)
    filename = f"{index:04d}_{museum_short}_{safe_name}{ext}"
    filepath = OUTPUT_DIR / filename

    result = {
        "index": index,
        "name": item["name"],
        "museum": item["museum"],
        "image_url": url,
        "detail_url": item.get("detail_url", ""),
        "description": item.get("description", ""),
        "local_path": str(filepath.relative_to(PROJECT_ROOT)),
        "status": "pending",
        "error": None,
    }

    # 如果已下载过且文件存在，跳过
    if filepath.exists() and filepath.stat().st_size > 0:
        result["status"] = "exists"
        return result

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": item.get("detail_url", url),
            }
            resp = requests.get(url, headers=headers, timeout=TIMEOUT, stream=True)
            resp.raise_for_status()

            # 校验返回的是图片
            content_type = resp.headers.get("Content-Type", "")
            if "image" not in content_type and "octet-stream" not in content_type:
                raise ValueError(f"Non-image content type: {content_type}")

            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            # 校验文件大小（小于1KB可能是错误页面）
            if filepath.stat().st_size < 1024:
                filepath.unlink()
                raise ValueError(f"File too small: {filepath.stat().st_size} bytes")

            result["status"] = "ok"
            return result

        except Exception as e:
            result["error"] = f"attempt {attempt}: {str(e)}"
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    result["status"] = "failed"
    # 清理可能的残留文件
    if filepath.exists():
        filepath.unlink()
    return result


def load_all_items() -> list[dict]:
    """从 enriched JSON 文件加载所有藏品，添加全局索引"""
    all_items = []
    global_index = 0

    for src_file in SOURCE_FILES:
        if not src_file.exists():
            print(f"[WARN] 文件不存在，跳过: {src_file}")
            continue

        with open(src_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"[INFO] 加载 {src_file.name}: {len(data)} 条")

        for item in data:
            if not item.get("image_url"):
                continue
            all_items.append({
                "index": global_index,
                "name": item["name"],
                "museum": item["museum"],
                "image_url": item["image_url"],
                "detail_url": item.get("detail_url", ""),
                "description": item.get("description", ""),
            })
            global_index += 1

    return all_items


def load_failed_items() -> list[dict]:
    """加载之前下载失败的条目，用于重试"""
    if not FAILED_FILE.exists():
        print("[WARN] 没有找到失败记录文件")
        return []

    with open(FAILED_FILE, "r", encoding="utf-8") as f:
        failed = json.load(f)

    print(f"[INFO] 加载 {len(failed)} 条失败记录，准备重试")
    return failed


def main():
    parser = argparse.ArgumentParser(description="批量下载博物馆藏品图片")
    parser.add_argument("--retry", action="store_true", help="重试之前失败的下载")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="并发线程数")
    args = parser.parse_args()

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    if args.retry:
        items = load_failed_items()
        if not items:
            return
    else:
        items = load_all_items()

    print(f"[INFO] 共 {len(items)} 条待下载")
    print(f"[INFO] 输出目录: {OUTPUT_DIR}")
    print(f"[INFO] 并发线程数: {args.workers}")
    print()

    # 并发下载
    results = []
    ok_count = 0
    skip_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, item): item for item in items}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)

            status = result["status"]
            if status == "ok":
                ok_count += 1
                tag = "OK"
            elif status == "exists":
                skip_count += 1
                tag = "SKIP"
            else:
                fail_count += 1
                tag = "FAIL"

            # 进度输出
            if i % 20 == 0 or status == "failed":
                print(
                    f"[{i}/{len(items)}] [{tag}] {result['index']:04d} {result['name']}"
                    + (f"  -- {result['error']}" if status == "failed" else "")
                )

    # 保存映射文件（全量，包含成功和已存在的）
    mapping = [r for r in results if r["status"] in ("ok", "exists")]
    # 按 index 排序
    mapping.sort(key=lambda x: x["index"])

    # 如果是重试模式，合并已有的 mapping
    if args.retry and MAPPING_FILE.exists():
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing_indices = {m["index"] for m in existing}
        for m in mapping:
            if m["index"] not in existing_indices:
                existing.append(m)
        mapping = sorted(existing + [m for m in mapping if m["index"] in existing_indices],
                         key=lambda x: x["index"])
        # 去重
        seen = set()
        deduped = []
        for m in mapping:
            if m["index"] not in seen:
                seen.add(m["index"])
                deduped.append(m)
        mapping = deduped

    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # 保存失败记录
    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        with open(FAILED_FILE, "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)

    # 汇总
    print()
    print("=" * 50)
    print(f"下载完成!")
    print(f"  成功: {ok_count}")
    print(f"  跳过(已存在): {skip_count}")
    print(f"  失败: {fail_count}")
    print(f"  映射文件: {MAPPING_FILE}")
    if failed:
        print(f"  失败记录: {FAILED_FILE}")
        print(f"  可用 --retry 重试失败项")
    print("=" * 50)


if __name__ == "__main__":
    main()
