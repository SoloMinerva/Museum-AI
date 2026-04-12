"""
中国国家博物馆藏品抓取（官网 https://www.chnmuseum.cn）

列表页：/zp/zpml/index.shtml（共 67 页，1336 件）
详情页：/zp/zpml/YYYYMM/tYYYYMMDD_XXXXX.shtml
分页命名：index.shtml → index_1.shtml → index_2.shtml ...

用法（项目根目录）:
  python -m src.utils.chnmuseum_fetch
  python -m src.utils.chnmuseum_fetch --pages 5 --out chnmuseum_sample.json
  python -m src.utils.chnmuseum_fetch --all --out chnmuseum_all.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.chnmuseum.cn"
LIST_BASE = f"{BASE_URL}/zp/zpml/"

HEADERS = {
    "User-Agent": "Museum-AI-Fetcher/1.0 (educational; +https://github.com/)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

REQUEST_DELAY = 0.8

MUSEUM_NAME = "中国国家博物馆"

# 详情页相对链接模式：./YYYYMM/tYYYYMMDD_XXXXX.shtml 或 ./kgfjp/YYYYMM/...
DETAIL_RE = re.compile(r"\./(?:[\w]+/)?(\d{6})/t(\d{8}_\d+)\.shtml")


def list_url(page: int) -> str:
    """第 0 页 → index.shtml，第 1 页 → index_1.shtml ..."""
    if page <= 0:
        return f"{LIST_BASE}index.shtml"
    return f"{LIST_BASE}index_{page}.shtml"


def _absolute_url(href: str, base: str = LIST_BASE) -> str:
    if not href or href.startswith("javascript:"):
        return ""
    if href.startswith("http"):
        return href
    return urljoin(base, href)


def _is_logo_or_ui_image(src: str) -> bool:
    s = (src or "").lower()
    return any(k in s for k in ("logo", "favicon", ".ico", "icon_", "/images/",
                                 "banner", "qrcode", "ewm"))


def fetch_list_page(page: int) -> list[dict]:
    """抓取藏品目录某一页，收集所有详情链接。"""
    url = list_url(page)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    items: list[dict] = []
    seen: set[str] = set()

    for a_tag in soup.select("a[href]"):
        href = a_tag.get("href", "")
        m = DETAIL_RE.search(href)
        if not m:
            continue
        item_id = m.group(2)  # YYYYMMDD_XXXXX
        if item_id in seen:
            continue
        seen.add(item_id)

        # 名称：取 a 标签内的文字，或 img 的 alt
        name = a_tag.get_text(strip=True)
        if not name:
            img = a_tag.find("img")
            if img:
                name = (img.get("alt") or "").strip()
        if not name or len(name) < 2:
            name = f"藏品_{item_id}"

        # 缩略图
        thumb = ""
        img_tag = a_tag.find("img")
        if img_tag:
            src = img_tag.get("src") or ""
            if src and not _is_logo_or_ui_image(src):
                thumb = _absolute_url(src)

        detail_url = _absolute_url(href)
        if not detail_url:
            continue

        # 时代：列表页通过 JS 替换分号为 ·，原始文本可能在相邻元素
        era = ""
        parent_li = a_tag.find_parent("li")
        if parent_li:
            # 寻找时代文本（如"西周""清"等）
            for span in parent_li.select("span, p, div"):
                t = span.get_text(strip=True)
                if t and t != name and len(t) < 30 and not t.startswith("http"):
                    era = t
                    break

        items.append({
            "id": item_id,
            "name": name,
            "era": era,
            "museum": MUSEUM_NAME,
            "image_url": thumb,
            "detail_url": detail_url,
        })

    # 去重：同一链接可能出现两次（图片 + 文字）
    deduped: list[dict] = []
    seen_urls: set[str] = set()
    for item in items:
        if item["detail_url"] not in seen_urls:
            seen_urls.add(item["detail_url"])
            deduped.append(item)
    return deduped


def fetch_detail(item: dict) -> dict:
    """抓取详情页，填充描述、属性、高清图。"""
    url = item["detail_url"]
    resp = None
    for attempt in range(2):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=25, allow_redirects=True)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 1:
                print(f"  [WARN] 详情失败 {item['name']}: {e}", file=sys.stderr)
                return item
            time.sleep(2)
    if resp is None:
        return item

    resp.encoding = resp.apparent_encoding or "utf-8"
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    # --- 名称 ---
    # 优先从 title 取
    if soup.title and soup.title.string:
        raw_title = soup.title.string.strip()
        title_name = raw_title
        for sep in ("-", "_", "|", "—"):
            if sep in raw_title:
                title_name = raw_title.split(sep)[0].strip()
                break
        if title_name and "国家博物馆" not in title_name and "中国" not in title_name and len(title_name) < 50:
            item["name"] = title_name
    # h2/h1 补充
    for sel in ("h2", "h1"):
        el = soup.find(sel)
        if el:
            t = el.get_text(strip=True)
            if t and "博物馆" not in t and len(t) < 60:
                item["name"] = t
                break

    # --- 年代（JS 变量 content2） ---
    era_match = re.search(r"var\s+content2\s*=\s*['\"](.+?)['\"]", html)
    if era_match and era_match.group(1).strip():
        item["era"] = era_match.group(1).strip()

    # --- 描述文字 ---
    texts: list[str] = []
    # 国博详情页描述在 .cj_dycp_rig 容器内的 p 标签中
    desc_container = soup.select_one(".cj_dycp_rig")
    if desc_container:
        for p in desc_container.select("p"):
            t = p.get_text(strip=True)
            if t and len(t) > 2:
                texts.append(t)

    # 兜底：全页 p 标签
    if not texts:
        for p in soup.find_all("p"):
            t = p.get_text(strip=True)
            if (t and len(t) > 8
                    and "分享" not in t and "备案" not in t
                    and "版权" not in t and "Copyright" not in t):
                texts.append(t)

    if texts:
        item["raw_description"] = "\n".join(texts)

    # --- 尺寸（从描述中提取第一行包含"厘米/cm"的） ---
    for t in texts:
        if "厘米" in t or "cm" in t.lower():
            item["size"] = t
            break

    # --- 材质（从描述中猜测） ---
    material_match = re.search(r"var\s+content\s*=\s*['\"](.+?)['\"]", html)
    if material_match and material_match.group(1).strip():
        item["material"] = material_match.group(1).strip()

    # --- 图片 ---
    # 优先取 .cj_small_img / #imgContainer 中的大图
    img_src = ""
    for container_sel in (".cj_small_img", "#imgContainer", ".zoomableContainer",
                          ".cj_big_mga"):
        container = soup.select_one(container_sel)
        if not container:
            continue
        for im in container.select("img"):
            src = im.get("src") or im.get("data-src") or ""
            if src and not _is_logo_or_ui_image(src):
                img_src = _absolute_url(src, base=url)
                break
        if img_src:
            break

    # 兜底：页面内最大可能是藏品图的 img
    if not img_src:
        for im in soup.select("img"):
            src = im.get("src") or ""
            if src and not _is_logo_or_ui_image(src) and src.endswith((".jpg", ".jpeg", ".png")):
                img_src = _absolute_url(src, base=url)
                break

    if img_src:
        item["image_url"] = img_src

    item["museum"] = MUSEUM_NAME
    return item


def to_kb_row(item: dict) -> dict:
    """转换为知识库导入格式。"""
    parts = []
    parts.append(f"收藏单位：{item.get('museum', MUSEUM_NAME)}")
    if item.get("era"):
        parts.append(f"年代：{item['era']}")
    if item.get("material"):
        parts.append(f"材质：{item['material']}")
    if item.get("size"):
        parts.append(f"尺寸：{item['size']}")

    meta_block = "\n".join(parts)
    raw_desc = item.get("raw_description", "")
    if raw_desc:
        description = f"{meta_block}\n\n{raw_desc}" if meta_block else raw_desc
    else:
        description = meta_block if meta_block else item.get("name", "")

    return {
        "name": item.get("name", ""),
        "museum": item.get("museum", MUSEUM_NAME),
        "description": description.strip(),
        "image_url": item.get("image_url", ""),
        "detail_url": item.get("detail_url", ""),
    }


def run(max_pages: int, fetch_all: bool = False) -> list[dict]:
    all_items: list[dict] = []
    page = -1
    while True:
        page += 1
        if not fetch_all and page >= max_pages:
            break
        label = f"[{page + 1}/{'∞' if fetch_all else max_pages}]"
        print(f"{label} 列表页 {page} ...", file=sys.stderr)
        try:
            items = fetch_list_page(page)
        except requests.RequestException as e:
            print(f"  [ERROR] 列表失败: {e}", file=sys.stderr)
            if fetch_all:
                break
            continue
        if not items:
            print(f"  第 {page} 页无藏品链接，停止。", file=sys.stderr)
            break
        print(f"  本页 {len(items)} 条（累计 {len(all_items) + len(items)}）", file=sys.stderr)
        for i, item in enumerate(items):
            print(f"  [{i + 1}/{len(items)}] {item['name']}", file=sys.stderr)
            time.sleep(REQUEST_DELAY)
            fetch_detail(item)
            all_items.append(item)
        time.sleep(REQUEST_DELAY)
    return all_items


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取中国国家博物馆藏品")
    parser.add_argument("--pages", type=int, default=2, help="列表分页数（默认 2）")
    parser.add_argument("--all", action="store_true", help="爬取全部 67 页")
    parser.add_argument("--out", type=str, default="chnmuseum_sample.json", help="输出 JSON")
    args = parser.parse_args()

    rows = run(args.pages, fetch_all=args.all)
    if not rows:
        print("未获取到任何数据。", file=sys.stderr)
        sys.exit(1)

    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    kb_path = out_path.replace(".json", "_kb_import.json")
    kb_rows = [to_kb_row(r) for r in rows]
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb_rows, f, ensure_ascii=False, indent=2)

    print(f"\n完成！共 {len(rows)} 条。", file=sys.stderr)
    print(f"  原始: {out_path}", file=sys.stderr)
    print(f"  KB:   {kb_path}", file=sys.stderr)
    print("\n--- 前 2 条 KB 预览 ---")
    print(json.dumps(kb_rows[:2], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
