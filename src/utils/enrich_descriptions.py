"""
文物描述 LLM 批量增强脚本

读取爬虫产出的 *_kb_import.json，对 description 过短的条目调用 LLM 生成
丰富的文物介绍文本，使其适合 LightRAG 实体/关系抽取。

用法（项目根目录）:
  python -m src.utils.enrich_descriptions --input sxhm_kb_import.json --out sxhm_kb_enriched.json
  python -m src.utils.enrich_descriptions --input hbkgy_kb_import.json --threshold 80

可选参数:
  --input      输入的 KB import JSON 文件路径（必需）
  --out        输出文件路径（默认在原文件名后加 _enriched）
  --threshold  描述长度阈值，低于此值的条目将被增强（默认 60 字符）
  --batch      每批并发请求数（默认 5）
  --delay      批次间延迟秒数（默认 1.0）
  --dry-run    仅统计需要增强的条目数，不实际调用 LLM
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import httpx
from dotenv import load_dotenv

# 加载项目 .env
load_dotenv()

# LLM API 配置：支持 DeepSeek / Ollama / SiliconFlow
API_KEY = os.getenv("ENRICH_API_KEY", os.getenv("SILICONFLOW_API_KEY", ""))
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"
DEEPSEEK_BASE = "https://api.deepseek.com/v1"

# 后端选择：deepseek / local / siliconflow
BACKEND = os.getenv("ENRICH_BACKEND", "siliconflow").lower()
if BACKEND == "deepseek":
    API_BASE = DEEPSEEK_BASE
    MODEL = os.getenv("ENRICH_LLM_MODEL", "deepseek-chat")
elif BACKEND == "local":
    API_BASE = OLLAMA_BASE
    MODEL = os.getenv("ENRICH_LLM_MODEL", "qwen3:8b")
else:
    API_BASE = SILICONFLOW_BASE
    MODEL = os.getenv("ENRICH_LLM_MODEL", "deepseek-ai/DeepSeek-V3")
USE_LOCAL = BACKEND == "local"

SYSTEM_PROMPT = """\
你是一位博物馆文物编目人员。请根据用户提供的字段信息，写一段简洁的文物简介（80~120字）。

写法要求：
1. 可以从文物名称中提取明显信息（如"玉牙璋"→材质为玉，器型为璋；"铜镜"→青铜质地的镜子）。
2. 可以根据器物类型补充一句通用的功能或用途说明（用"通常""一般"限定）。
3. 将名称、年代、类别、材质、尺寸、原始描述等信息用通顺的语言整合为完整段落。
4. 不要重复博物馆名称。
5. 只输出正文，不加标题或标签。

禁止事项：
- 禁止编造原始信息中没有的具体事实（如具体纹饰、出土地点、工艺细节）。
- 禁止出现"信息不详""暂无信息""尺寸不详""年代不详"等表述，缺少的信息直接跳过不提。
- 禁止写套话（如"需进一步研究""有待考证"）。
- 禁止写总结性评价（如"体现了……""具有重要价值""精湛工艺"）。
- 如果可用信息极少，至少根据文物名称推断材质和类别写一句说明，不要重复博物馆名称。"""


CATEGORY_LABEL = {
    "qtq": "青铜器", "zgzb": "镇馆之宝", "qmq": "漆木器",
    "jyq": "金银器", "zhujian": "竹木简牍", "yuqi": "玉器",
    "tcq": "陶瓷器", "shuhua": "书画", "gjsb": "古籍善本",
    "gymsjp": "工艺美术精品",
}


def build_user_prompt(item: dict) -> str:
    """根据 KB 条目构造 LLM 用户提示，直接嵌入各字段值。"""
    # 从 _raw 原始数据中提取结构化字段（如果有的话）
    raw = item.get("_raw", {})
    name = item.get("name", "")
    museum = item.get("museum", "")
    era = raw.get("era", "") or ""
    category_raw = raw.get("category", "") or ""
    category = CATEGORY_LABEL.get(category_raw, category_raw)
    material = raw.get("material", "") or ""
    raw_desc = raw.get("raw_description", "") or item.get("description", "")

    parts = []
    if USE_LOCAL:
        parts.append("/no_think")
    parts += [
        f"文物名称：{name}",
        f"博物馆：{museum}",
        f"年代：{era if era else '未知'}",
        f"类别：{category if category else '未知'}",
        f"材质：{material if material else '未知'}",
        f"原始描述：{raw_desc if raw_desc else '无'}",
    ]
    return "\n".join(parts)


async def call_llm(client: httpx.AsyncClient, user_prompt: str) -> str:
    """调用 LLM API（Ollama 或 SiliconFlow）。"""
    request_body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1000,
    }
    # Ollama 本地模式：关闭 thinking 以节省 token
    if USE_LOCAL:
        request_body["think"] = False

    resp = await client.post(
        f"{API_BASE}/chat/completions",
        json=request_body,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    # 兜底：如果仍有 <think> 标签，去除
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    return content


async def enrich_batch(
    client: httpx.AsyncClient,
    items: list[dict],
    indices: list[int],
    threshold: int,
) -> int:
    """对一批条目并发调用 LLM 增强描述，返回成功数。"""
    tasks = []
    for idx in indices:
        item = items[idx]
        prompt = build_user_prompt(item)
        tasks.append(call_llm(client, prompt))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    success = 0
    for idx, result in zip(indices, results):
        item = items[idx]
        if isinstance(result, Exception):
            print(f"  [WARN] 增强失败 {item['name']}: {result}", file=sys.stderr)
            continue
        # 将 LLM 生成的介绍追加到原有 description 后面
        old_desc = item.get("description", "")
        item["description"] = f"{old_desc}\n\n{result}".strip()
        success += 1
    return success


async def enrich_all(
    items: list[dict],
    threshold: int = 60,
    batch_size: int = 5,
    delay: float = 1.0,
) -> int:
    """增强所有描述过短的条目，返回增强总数。"""
    # 找出需要增强的索引
    to_enrich = [
        i for i, item in enumerate(items)
        if len(item.get("description", "")) < threshold
    ]
    if not to_enrich:
        print("所有条目描述已足够丰富，无需增强。", file=sys.stderr)
        return 0

    print(f"共 {len(items)} 条，需增强 {len(to_enrich)} 条（阈值 {threshold} 字符）", file=sys.stderr)

    headers = {"Content-Type": "application/json"}
    if not USE_LOCAL and API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    total_success = 0
    async with httpx.AsyncClient(headers=headers) as client:
        for batch_start in range(0, len(to_enrich), batch_size):
            batch_indices = to_enrich[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(to_enrich) + batch_size - 1) // batch_size
            print(
                f"  批次 [{batch_num}/{total_batches}] "
                f"处理 {len(batch_indices)} 条...",
                file=sys.stderr,
            )
            success = await enrich_batch(client, items, batch_indices, threshold)
            total_success += success
            if batch_start + batch_size < len(to_enrich):
                await asyncio.sleep(delay)

    return total_success


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 批量增强文物描述")
    parser.add_argument("--input", type=str, required=True, help="输入 KB import JSON")
    parser.add_argument("--raw", type=str, default="", help="原始爬虫 JSON（提供 era/category/material 等字段）")
    parser.add_argument("--out", type=str, default="", help="输出文件路径")
    parser.add_argument("--threshold", type=int, default=60, help="描述长度阈值（字符）")
    parser.add_argument("--batch", type=int, default=5, help="每批并发数")
    parser.add_argument("--delay", type=float, default=1.0, help="批次间延迟（秒）")
    parser.add_argument("--dry-run", action="store_true", help="仅统计，不调用 LLM")
    args = parser.parse_args()

    if not USE_LOCAL and not API_KEY:
        print("错误：未设置 ENRICH_API_KEY 环境变量", file=sys.stderr)
        sys.exit(1)
    print(f"使用模型: {MODEL} ({BACKEND})", file=sys.stderr)

    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)

    # 加载原始爬虫数据，按 name 匹配挂载到 _raw 字段
    if args.raw:
        with open(args.raw, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
        raw_by_name = {}
        for r in raw_list:
            raw_by_name[r.get("name", "")] = r
        matched = 0
        for item in items:
            raw = raw_by_name.get(item.get("name", ""))
            if raw:
                item["_raw"] = raw
                matched += 1
        print(f"原始数据匹配: {matched}/{len(items)} 条", file=sys.stderr)
    else:
        print("未提供 --raw，将仅从 KB 字段构造 prompt", file=sys.stderr)

    print(f"读取 {len(items)} 条记录", file=sys.stderr)

    # 统计当前描述质量
    short_count = sum(1 for it in items if len(it.get("description", "")) < args.threshold)
    print(f"描述低于 {args.threshold} 字符的条目: {short_count}/{len(items)}", file=sys.stderr)

    if args.dry_run:
        # 预览前 3 条需要增强的
        need = [it for it in items if len(it.get("description", "")) < args.threshold]
        for it in need[:3]:
            print(f"\n  名称: {it['name']}")
            print(f"  当前描述: {it.get('description', '')[:100]}...")
        print(f"\n总计 {short_count} 条需要增强（dry-run 模式，未调用 LLM）")
        return

    enriched = asyncio.run(
        enrich_all(items, args.threshold, args.batch, args.delay)
    )

    out_path = args.out or args.input.replace(".json", "_enriched.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"\n完成！增强 {enriched} 条，输出: {out_path}", file=sys.stderr)

    # 预览
    print("\n--- 前 2 条增强后预览 ---")
    enriched_items = [it for it in items if len(it.get("description", "")) >= args.threshold]
    print(json.dumps(enriched_items[:2], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
