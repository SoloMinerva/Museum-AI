"""
从全量文物数据中抽样 300-500 条，覆盖不同年代、类别、收藏单位。
按比例分层抽样，确保各维度均有代表。

输出: saves/knowledge_base_data/lightrag_data/combined_museum_sampled.txt
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "saves" / "knowledge_base_data" / "lightrag_data" / "combined_museum_all_chunked.txt"
OUTPUT_FILE = PROJECT_ROOT / "saves" / "knowledge_base_data" / "lightrag_data" / "combined_museum_sampled.txt"

TARGET_COUNT = 400  # 目标抽样数
random.seed(42)


def extract_metadata(entry: str) -> dict:
    """从条目中提取元数据"""
    meta = {}

    # 收藏单位
    m = re.search(r"收藏单位[：:]\s*(.+)", entry)
    meta["museum"] = m.group(1).strip() if m else "未知"

    # 类别
    m = re.search(r"类别[：:]\s*(.+)", entry)
    meta["category"] = m.group(1).strip() if m else "未知"

    # 年代
    m = re.search(r"年代[：:]\s*(.+)", entry)
    if m:
        raw = m.group(1).strip().split(";")[0].split("；")[0].strip()
    else:
        raw = ""

    # 归一化年代到大类
    period_map = [
        ("新石器", "新石器"),
        ("商", "商"),
        ("西周", "西周"),
        ("东周", "东周"),
        ("春秋", "春秋"),
        ("战国", "战国"),
        ("秦", "秦"),
        ("汉", "汉"),
        ("三国", "三国"),
        ("晋", "晋"),
        ("南北朝", "南北朝"),
        ("隋", "隋"),
        ("唐", "唐"),
        ("宋", "宋"),
        ("元", "元"),
        ("明", "明"),
        ("清", "清"),
        ("近代", "近现代"),
        ("现当代", "近现代"),
        ("民国", "近现代"),
    ]
    period = "未知"
    search_text = raw + entry[:300]
    for keyword, label in period_map:
        if keyword in search_text:
            period = label
            break
    meta["period"] = period

    # 名称
    m = re.search(r"【(.+?)】", entry)
    meta["name"] = m.group(1) if m else ""

    return meta


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    entries = [e.strip() for e in text.split("---") if e.strip()]
    print(f"总条目数: {len(entries)}")

    # 提取每条的元数据
    all_items = []
    for i, entry in enumerate(entries):
        meta = extract_metadata(entry)
        meta["index"] = i
        meta["text"] = entry
        all_items.append(meta)

    # 统计分布
    print("\n原始分布:")
    for dim in ["museum", "category", "period"]:
        counts = {}
        for item in all_items:
            k = item[dim]
            counts[k] = counts.get(k, 0) + 1
        print(f"  {dim}: {dict(sorted(counts.items(), key=lambda x: -x[1]))}")

    # 分层抽样：国博多选（质量高，平均354字），湖北省博少选（平���115字）
    # 目标：国博 ~260 条，湖北省博 ~140 条
    museum_quota = {}
    for item in all_items:
        museum_quota.setdefault(item["museum"], 0)
    gj_total = sum(1 for item in all_items if "国家" in item["museum"])
    hb_total = len(all_items) - gj_total
    gj_target = min(260, gj_total)  # 国博尽量多选
    hb_target = TARGET_COUNT - gj_target

    # 按 (museum, period) 组合分组
    groups = {}
    for item in all_items:
        key = (item["museum"], item["period"])
        groups.setdefault(key, []).append(item)

    sampled = []
    for key, items in groups.items():
        is_gj = "国家" in key[0]
        pool_total = gj_total if is_gj else hb_total
        target = gj_target if is_gj else hb_target
        n = max(1, round(len(items) / pool_total * target))
        n = min(n, len(items))
        chosen = random.sample(items, n)
        sampled.extend(chosen)

    # 调整到目标数
    if len(sampled) > TARGET_COUNT:
        sampled = random.sample(sampled, TARGET_COUNT)

    if len(sampled) < TARGET_COUNT:
        selected_indices = {item["index"] for item in sampled}
        # 优先从国博补
        pool_gj = [item for item in all_items if item["index"] not in selected_indices and "国家" in item["museum"]]
        pool_hb = [item for item in all_items if item["index"] not in selected_indices and "国家" not in item["museum"]]
        pool = pool_gj + pool_hb
        extra = random.sample(pool, min(TARGET_COUNT - len(sampled), len(pool)))
        sampled.extend(extra)

    # 按原始顺序排列
    sampled.sort(key=lambda x: x["index"])

    print(f"\n抽样结果: {len(sampled)} 条")

    # 统计抽样后分布
    print("\n抽样后分布:")
    for dim in ["museum", "category", "period"]:
        counts = {}
        for item in sampled:
            k = item[dim]
            counts[k] = counts.get(k, 0) + 1
        print(f"  {dim}: {dict(sorted(counts.items(), key=lambda x: -x[1]))}")

    # 输出
    output_text = "\n\n---\n\n".join(item["text"] for item in sampled)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output_text)

    size = OUTPUT_FILE.stat().st_size
    print(f"\n输出文件: {OUTPUT_FILE.name} ({size} bytes)")
    print(f"覆盖文物示例:")
    for item in sampled[:5]:
        print(f"  [{item['period']}] {item['name']} ({item['museum']}, {item['category']})")
    print(f"  ...")
    for item in sampled[-3:]:
        print(f"  [{item['period']}] {item['name']} ({item['museum']}, {item['category']})")


if __name__ == "__main__":
    main()
