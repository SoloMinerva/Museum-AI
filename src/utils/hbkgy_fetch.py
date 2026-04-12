"""
湖北省博物馆典藏抓取（列表托管在长江云：http://hbsbwg.cjyun.org）

官网入口见 https://www.hbkgy.com/home/index.html ，典藏详情页多为 /p/{id}.html 形式。
结构与 tjbwg_fetch 一致：列表（支持分页）→ 详情 → 原始 JSON + *_kb_import.json

用法（项目根目录）:
  python -m src.utils.hbkgy_fetch
  python -m src.utils.hbkgy_fetch --category qtq --pages 3 --out hubei_qtq.json
  python -m src.utils.hbkgy_fetch --all --out hbkgy_all.json

可选参数:
  --category   典藏子路径：qtq 青铜器 zgzb 镇馆之宝 qmq 漆木器 等（默认 qtq）
  --pages      列表分页数（第 1 页 index.html，其后 index_2.html ...，默认 2）
  --all        自动爬取所有分类的所有页面
  --out        输出文件路径（默认 hbkgy_sample.json）
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

# 典藏内容实际域名（与 hbkgy.com 导航一致）
BASE_URL = "http://hbsbwg.cjyun.org"

HEADERS = {
    "User-Agent": "Museum-AI-Fetcher/1.0 (educational; +https://github.com/)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

REQUEST_DELAY = 0.8

# 详情页路径模式：/p/4885.html
P_DETAIL_RE = re.compile(r"/p/(\d+)\.html", re.I)

# 全站导航里反复出现的概况/资讯类页面 ID，避免当成藏品
EXCLUDE_P_IDS = {
    "4885",
    "7900",
}

# 列表路径 → 可读门类（写入 description 元数据行）
CATEGORY_LABEL = {
    "qtq": "青铜器",
    "zgzb": "镇馆之宝",
    "qmq": "漆木器",
    "jyq": "金银器",
    "zhujian": "竹木简牍",
    "yuqi": "玉器",
    "tcq": "陶瓷器",
    "shuhua": "书画",
    "gjsb": "古籍善本",
    "gymsjp": "工艺美术精品",
}


MUSEUM_NAME = "湖北省博物馆"


def _is_logo_or_ui_image(src: str) -> bool:
    s = (src or "").lower()
    if "logo" in s or "favicon" in s or s.endswith(".ico"):
        return True
    # 排除站点 UI 资源（assets 目录下的图标/装饰图）
    if "/assets/" in s and "img.cjyun.org" not in s:
        return True
    if "icon_hzw" in s:
        return True
    return False


def list_url(category: str, page: int) -> str:
    cat = category.strip().strip("/")
    if page <= 1:
        return f"{BASE_URL}/{cat}/index.html"
    return f"{BASE_URL}/{cat}/index_{page}.html"


def _absolute_url(href: str) -> str:
    if not href or href.startswith("javascript:"):
        return ""
    return href if href.startswith("http") else urljoin(BASE_URL, href)


def fetch_list_page(category: str, page: int) -> list[dict]:
    """抓取某一典藏分类的一页列表，收集所有 /p/{id}.html 链接。"""
    url = list_url(category, page)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")
    items: list[dict] = []
    seen: set[str] = set()

    # 实际页面结构：#mainlist > ul > li，每个 li 包含：
    #   <a class="list_photo" href="/qtq/p/5522.html"><img src="..."></a>
    #   <div class="list_text"><a href="..."><h4>铜镜</h4></a></div>
    mainlist = soup.select_one("#mainlist")
    if mainlist:
        for li in mainlist.select("li"):
            # 从 list_text 区域取名称
            text_a = li.select_one(".list_text a")
            if not text_a:
                continue
            href = text_a.get("href") or ""
            m = P_DETAIL_RE.search(href)
            if not m:
                continue
            item_id = m.group(1)
            if item_id in EXCLUDE_P_IDS or item_id in seen:
                continue
            seen.add(item_id)

            name = text_a.get_text(strip=True)
            if len(name) < 2:
                name = f"文物_{item_id}"
            if name in ("馆长致辞", "馆情简介", "首页", "概况", "资讯"):
                continue

            # 从 list_photo 区域取缩略图
            photo_img = li.select_one(".list_photo img")
            thumb_url = ""
            if photo_img:
                src = photo_img.get("src") or photo_img.get("data-src") or ""
                if src and not _is_logo_or_ui_image(src):
                    thumb_url = src if src.startswith("http") else _absolute_url(src)

            detail_href = _absolute_url(href)
            if not detail_href:
                continue

            items.append({
                "id": item_id,
                "name": name,
                "category": category,
                "museum": MUSEUM_NAME,
                "era": "",
                "image_url": thumb_url,
                "detail_url": detail_href,
            })

    # 兜底：如果 #mainlist 不存在，回退到全页扫描
    if not items:
        for a in soup.select('a[href*="/p/"]'):
            href = a.get("href") or ""
            m = P_DETAIL_RE.search(href)
            if not m:
                continue
            item_id = m.group(1)
            if item_id in EXCLUDE_P_IDS or item_id in seen:
                continue
            seen.add(item_id)
            name = a.get_text(strip=True)
            if len(name) < 2:
                name = f"文物_{item_id}"
            if name in ("馆长致辞", "馆情简介", "首页", "概况", "资讯"):
                continue
            detail_url = _absolute_url(href)
            if not detail_url:
                continue
            items.append({
                "id": item_id,
                "name": name,
                "category": category,
                "museum": MUSEUM_NAME,
                "era": "",
                "image_url": "",
                "detail_url": detail_url,
            })

    return items


def fetch_detail(item: dict) -> dict:
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
    soup = BeautifulSoup(resp.text, "html.parser")

    # 标题：<title>「铜镜 - 湖北省博物馆」、或正文 h1/h2
    if soup.title and soup.title.string:
        head = soup.title.string.strip()
        if "-" in head:
            cand = head.split("-")[0].strip()
            if cand and "湖北省博物馆" not in cand:
                item["name"] = cand
        elif head and "湖北省博物馆" not in head:
            item["name"] = head
    for sel in ("h1", "h2"):
        el = soup.find(sel)
        if not el:
            continue
        t = el.get_text(strip=True)
        if t and "湖北省博物馆" not in t and len(t) < 100:
            item["name"] = t
            break

    main = (
        soup.select_one(".TRS_Editor")
        or soup.select_one("#main")
        or soup.select_one("#content")
        or soup.select_one(".content")
        or soup.select_one("article")
    )
    texts: list[str] = []
    if main:
        for p in main.select("p"):
            t = p.get_text(strip=True)
            if t and len(t) > 2 and "分享到" not in t:
                texts.append(t)
    if not texts:
        for p in soup.find_all("p"):
            t = p.get_text(strip=True)
            if len(t) > 8 and "分享到" not in t and "备案" not in t:
                texts.append(t)

    # 优先从正文区域取文物图片（img.cjyun.org 域名为真实藏品图）
    img_src = ""
    scope = main or soup
    for im in scope.select("img"):
        src = im.get("src") or im.get("data-src") or ""
        if not src or _is_logo_or_ui_image(src):
            continue
        abs_src = src if src.startswith("http") else _absolute_url(src)
        # 优先选择 img.cjyun.org 的藏品图
        if "img.cjyun.org" in abs_src:
            img_src = abs_src
            break
        if not img_src:
            img_src = abs_src
    if img_src:
        item["image_url"] = img_src

    if texts:
        item["raw_description"] = "\n".join(texts)

    for row in soup.select("tr"):
        cells = [c.get_text(strip=True) for c in row.select("td,th")]
        if len(cells) >= 2:
            k, v = cells[0], cells[1]
            if "年代" in k and v:
                item["era"] = item.get("era") or v
            if "类别" in k and v:
                item["category"] = item.get("category") or v

    item["museum"] = MUSEUM_NAME
    return item


def to_kb_row(item: dict) -> dict:
    parts = []
    parts.append(f"收藏单位：{item.get('museum', MUSEUM_NAME)}")
    if item.get("era"):
        parts.append(f"年代：{item['era']}")
    cat_key = item.get("category") or ""
    cat_show = CATEGORY_LABEL.get(cat_key, cat_key)
    if cat_show:
        parts.append(f"类别：{cat_show}")
    if item.get("material"):
        parts.append(f"材质：{item['material']}")
    if item.get("level"):
        parts.append(f"级别：{item['level']}")
    if item.get("year_collected"):
        parts.append(f"入藏年度：{item['year_collected']}")

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


def run(category: str, max_pages: int, fetch_all: bool = False) -> list[dict]:
    all_items: list[dict] = []
    page = 0
    while True:
        page += 1
        if not fetch_all and page > max_pages:
            break
        label = f"[{page}/∞]" if fetch_all else f"[{page}/{max_pages}]"
        print(f"{label} 列表 {category} ...", file=sys.stderr)
        try:
            items = fetch_list_page(category, page)
        except requests.RequestException as e:
            print(f"  [ERROR] 列表失败: {e}", file=sys.stderr)
            continue
        if not items:
            print(f"  第 {page} 页无藏品链接，停止。", file=sys.stderr)
            break
        print(f"  本页 {len(items)} 条链接（累计 {len(all_items) + len(items)}）", file=sys.stderr)
        for i, item in enumerate(items):
            print(f"  [{i+1}/{len(items)}] {item['name']}", file=sys.stderr)
            time.sleep(REQUEST_DELAY)
            fetch_detail(item)
            all_items.append(item)
        time.sleep(REQUEST_DELAY)
    return all_items


def run_all_categories(max_pages: int, fetch_all: bool = True) -> list[dict]:
    """爬取所有分类的所有页面。"""
    all_items: list[dict] = []
    for cat_key, cat_label in CATEGORY_LABEL.items():
        print(f"\n{'='*50}", file=sys.stderr)
        print(f"开始爬取分类: {cat_label}（{cat_key}）", file=sys.stderr)
        print(f"{'='*50}", file=sys.stderr)
        items = run(cat_key, max_pages, fetch_all=fetch_all)
        all_items.extend(items)
        print(f"分类 {cat_label} 完成，获取 {len(items)} 条（总计 {len(all_items)}）", file=sys.stderr)
    return all_items


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取湖北省博物馆典藏（cjyun 源站）")
    parser.add_argument("--category", type=str, default="qtq", help="典藏路径段，如 qtq、zgzb、qmq")
    parser.add_argument("--pages", type=int, default=2, help="列表分页数")
    parser.add_argument("--all", action="store_true", help="爬取所有分类的所有页面")
    parser.add_argument("--out", type=str, default="hbkgy_sample.json", help="输出 JSON")
    args = parser.parse_args()

    if args.all:
        rows = run_all_categories(max_pages=args.pages, fetch_all=True)
    else:
        rows = run(args.category.strip("/"), args.pages)
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
    print(f"  KB: {kb_path}", file=sys.stderr)
    print("\n--- 前 2 条 KB 预览 ---")
    print(json.dumps(kb_rows[:2], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
