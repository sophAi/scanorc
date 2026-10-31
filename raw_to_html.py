#!/usr/bin/env python3
"""
ScanORC: OCR Raw 檔直接組裝 HTML / EPUB 工具 (raw_to_html.py)
由 raw_pages/ 目錄中的各頁純文字結構化 JSON 檔與 toc.json，直接快速生成語意結構化 HTML 與 EPUB。

特色：
  1. 極速輕量：無需 GPU 或 AI 模型（零 VRAM 佔用），可在 0.1~1 秒內完成整本書的 HTML / EPUB 組裝。
  2. 完美支援目錄索引 (TOC)：自動讀取 toc.json，生成章節錨點、點狀引導線與 EPUB Outline 標籤。
  3. 雙向排版相容：原生支援直排 (vertical-rl) 與橫排 (horizontal-tb) CSS 樣式與閱讀模式。
  4. 獨立指令與模組兼備：可作為獨立 CLI 指令執行，亦可供 pipeline.py 等模組直接呼叫。
"""

import os
import re
import sys
import json
import shutil
import zipfile
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Any

from config_utils import attach_preference_and_template_support

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def natural_sort_key(s: Union[str, Path]) -> List[Any]:
    """檔名自然數字排序鍵 (例如 Cover-05 排在 Cover-06 前面，Page-2 排在 Page-10 前面)"""
    name = Path(s).name if isinstance(s, Path) else str(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", name)]


def classify_book_image(p: Union[str, Path]) -> int:
    """
    分類書本頁面類型：
    0: 封面/目錄 (Cover, 封面, front_cover 等)
    1: 正文頁面 (Page-xxx, p-xxx, p0xx, 純數字檔名, 頁等)
    2: 其他/未知頁面 (其他插頁等)
    3: 封底/後記 (BackCover, 封底等)
    """
    stem = Path(p).stem.lower()
    if re.search(r"(back[-_]?cover|封底)", stem):
        return 3
    if re.search(r"(^cover|^封面|^front[-_]?cover|^content)", stem):
        return 0
    if re.search(r"(^page|^p[-_]?\d+|^\d+$|頁|页)", stem):
        return 1
    return 2


def book_page_sort_key(p: Union[str, Path], cover_first: bool = True) -> Tuple[int, List[Any], str]:
    """
    頁面全域排序鍵：
    預設編排順序：Cover-*.json (優先) -> Page-*.json -> 其他 -> BackCover
    若 cover_first 為 False：Page-*.json (優先) -> Cover-*.json -> 其他 -> BackCover
    """
    c_type = classify_book_image(p)
    if cover_first:
        priority_map = {0: 0, 1: 1, 2: 2, 3: 3}
    else:
        priority_map = {1: 0, 0: 1, 2: 2, 3: 3}
    priority = priority_map.get(c_type, 2)
    return (priority, natural_sort_key(p), str(p))


CIRCLED_CHARS_RE = r"[\u2460-\u2473\u24ea\u3251-\u325f\u32b1-\u32bf\u24d0-\u24e9\u24b6-\u24cf]"

# 允許合法保留的 HTML 標籤白名單 (不分大小寫)
ALLOWED_HTML_TAGS = {
    "html", "head", "body", "meta", "title", "style", "link",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p", "div", "span", "section", "hr", "br",
    "a", "img", "figure", "figcaption",
    "table", "thead", "tbody", "tr", "th", "td",
    "ul", "ol", "li",
    "b", "i", "em", "strong", "sup", "sub", "code", "pre", "small", "mark", "del", "ins", "blockquote"
}


def sanitize_html_tags(html_text: str) -> str:
    """
    清洗 HTML 字串，防止非 HTML 標籤之角括號 (如 C++ 模板 <Q>, <T>, <vector>, 或 <iostream> 等)
    被 Pandoc 或 HTML 解析器誤判為 HTML 標籤 (如 <Q> 被誤認為 <q> 引號標籤) 導致 TagClose 錯誤。
    """
    if not html_text:
        return ""

    def replace_tag(match):
        tag_str = match.group(0)
        # 註解保留
        if tag_str.startswith("<!--") and tag_str.endswith("-->"):
            return tag_str
        # 擷取標籤名稱 (包含開頭可選的 /)
        m_tag = re.match(r"^</?([a-zA-Z0-9_-]+)", tag_str)
        if m_tag:
            tag_name = m_tag.group(1).lower()
            if tag_name in ALLOWED_HTML_TAGS:
                return tag_str
        # 不在允許清單內的標籤，將角括號轉義為 HTML 實體
        return tag_str.replace("<", "&lt;").replace(">", "&gt;")

    return re.sub(r"<[^>]*>", replace_tag, html_text)


def to_circled_char(key: str) -> str:
    """將數字 (0~50) 或字母 (a~z, A~Z) 轉換為標準 Unicode 帶圈字元 (①, ②, ㉑, ㊱, ⓐ, Ⓐ 等)"""
    key = str(key).strip()
    # 若本身已是 Unicode 帶圈字符，直接原樣保留
    if re.match(r"^" + CIRCLED_CHARS_RE + r"$", key):
        return key

    # 僅純阿拉伯數字才嘗試轉換 (排除 Unicode 數字如 '③' 在 isdigit() 為 True 但 int() 報錯的情況)
    if key.isdecimal():
        try:
            n = int(key)
            if n == 0:
                return "⓪"
            elif 1 <= n <= 20:
                return chr(0x2460 + n - 1)
            elif 21 <= n <= 35:
                return chr(0x3251 + n - 21)
            elif 36 <= n <= 50:
                return chr(0x32B1 + n - 36)
            else:
                return f"({n})"
        except (ValueError, TypeError):
            return f"({key})"
    elif len(key) == 1:
        if "a" <= key <= "z":
            return chr(0x24D0 + ord(key) - ord("a"))
        elif "A" <= key <= "Z":
            return chr(0x24B6 + ord(key) - ord("A"))
    return f"({key})"


P_CIRCLED_UNWRAP = re.compile(r"(?<!\$)\$(?!\$)\s*(" + CIRCLED_CHARS_RE + r")\s*(?<!\$)\$(?!\$)")
P_CIRCLED_SUP1 = re.compile(r"(?<!\$)\$(?!\$)\s*\^\{\s*(" + CIRCLED_CHARS_RE + r")\s*\}\s*(?<!\$)\$(?!\$)")
P_CIRCLED_SUP2 = re.compile(r"(?<!\$)\$(?!\$)\s*\^(" + CIRCLED_CHARS_RE + r")\s*(?<!\$)\$(?!\$)")
P_WORD_SUP = re.compile(r"(?<!\$)\$(?!\$)\s*([A-Za-z0-9_\u4e00-\u9fa5]*)(?:\\mathrm\{([^{}]+)\})?\s*\^\{\s*(" + CIRCLED_CHARS_RE + r")\s*\}\s*(?<!\$)\$(?!\$)")


def clean_math_syntax(text: str) -> str:
    """
    清理並標準化 LaTeX 數學語法：
    1. 將 GLM-OCR 識別輸出之 \\textcircled{n} 自動轉譯為標準 Unicode 帶圈註腳符號 (①, ②, ③...)。
    2. 解除獨立帶圈數字的 $...$ 包裹 ($①$ -> ①, $^{①}$ -> <sup>①</sup>)，防止 Pandoc MathML 引擎解析失敗而殘留原始碼。
    3. 自動正規化註腳上標 ($^{5,6}$ -> <sup>5,6</sup>, $^{1}$ -> <sup>1</sup>, $2\\mathrm{ch}^{②}$ -> 2ch<sup>②</sup>)。
    4. 去除行內公式 $ ... $ 內部頭尾的空白字元（符合 Pandoc 與標準 TeX 語法）。
    5. 將公式末尾包含的句點或逗號移至公式外，例如 `$x.$` -> `$x$.`
    6. 保留非公式的貨幣金額（如 $100 到 $200），避免誤將貨幣符號當作公式處理。
    7. 標準化 \\( ... \\) 行內公式為 $ ... $。
    """
    if not text:
        return ""

    # 1. 將所有 \textcircled{...} 轉換為 Unicode 帶圈字元 (①, ② ...) (支援大小寫與空格容錯)
    text = re.sub(r"\\text[cC]irc(?:le|re)?d\s*\{([^{}]+)\}", lambda m: to_circled_char(m.group(1)), text)

    # 2. 處理以 $ 包裹的單一帶圈數字，直接解包為純字元: $①$ -> ①
    text = P_CIRCLED_UNWRAP.sub(r"\1", text)

    # 3. 處理以 $ 包裹的上標帶圈數字: $^{①}$ 或 $^{ ① }$ -> <sup>①</sup>
    text = P_CIRCLED_SUP1.sub(r"<sup>\1</sup>", text)
    text = P_CIRCLED_SUP2.sub(r"<sup>\1</sup>", text)

    # 4. 處理帶有詞彙的上標帶圈數字，例如 $2\mathrm{ch}^{ ② }$ -> 2ch<sup>②</sup>
    def fix_word_superscript(m):
        prefix = m.group(1) or ""
        math_rm = m.group(2) or ""
        sup = m.group(3) or ""
        word = prefix + math_rm
        return f"{word}<sup>{sup}</sup>"

    text = P_WORD_SUP.sub(fix_word_superscript, text)

    # 5. 處理純數字/逗號之上標註腳: $^{5,6}$ -> <sup>5,6</sup>, $^{1}$ -> <sup>1</sup>
    text = re.sub(r"(?<!\$)\$(?!\$)\s*\^\{\s*([0-9,，\s\-]+)\s*\}\s*(?<!\$)\$(?!\$)", lambda m: f"<sup>{m.group(1).strip()}</sup>", text)
    text = re.sub(r"(?<!\$)\$(?!\$)\s*\^([0-9]+)\s*(?<!\$)\$(?!\$)", r"<sup>\1</sup>", text)

    # 6. 標準化 \( ... \) 行內公式為 $ ... $
    def fix_paren_math(match):
        inner = match.group(1).strip()
        trailing_punct = ""
        if inner and inner[-1] in ".，,。；;":
            trailing_punct = inner[-1]
            inner = inner[:-1].strip()
        return f"${inner}${trailing_punct}" if inner else trailing_punct

    text = re.sub(r"\\\(\s*([^\n\\]+?)\s*\\\)", fix_paren_math, text)

    # 7. 去除行內公式 $ ... $ 內部頭尾的空白字元
    def fix_inline_math(match):
        inner = match.group(1).strip()
        trailing_punct = ""
        if inner and inner[-1] in ".，,。；;":
            trailing_punct = inner[-1]
            inner = inner[:-1].strip()
        return f"${inner}${trailing_punct}" if inner else trailing_punct

    text = re.sub(r"(?<!\$)\$(?!\$)\s*([^\$\n]+?)\s*(?<!\$)\$(?!\$|\d)", fix_inline_math, text)

    return text


def normalize_formula_content(content: str) -> str:
    """
    清理並標準化獨立公式 (isolate_formula) 區塊內容：
    1. 剝離可能多重巢狀包裹的定界符 (如 $$, \\[, \\(, $)
    2. 統一以標準的 $$ ... $$ 格式包覆，確保在 HTML (KaTeX) 與 EPUB (Pandoc MathML) 中均能 100% 正確渲染
    """
    if not content:
        return ""
    c = clean_math_syntax(content.strip())
    while True:
        stripped = False
        if c.startswith("$$") and c.endswith("$$") and len(c) >= 4:
            c = c[2:-2].strip()
            stripped = True
        elif (c.startswith(r"\[") or c.startswith("\\[")) and (c.endswith(r"\]") or c.endswith("\\]")) and len(c) >= 4:
            c = c[2:-2].strip()
            stripped = True
        elif (c.startswith(r"\(") or c.startswith("\\(")) and (c.endswith(r"\)") or c.endswith("\\)")) and len(c) >= 4:
            c = c[2:-2].strip()
            stripped = True
        elif c.startswith("$") and c.endswith("$") and len(c) >= 2 and not c.startswith("$$"):
            c = c[1:-1].strip()
            stripped = True
        if not stripped:
            break
    if not c:
        return ""
    return f"$$\n{c}\n$$"


def resolve_page_filename(page_num: int, existing_filenames: Optional[List[str]] = None) -> str:
    """
    根據頁碼推算目標圖檔檔名 (例如 50 -> Page-050.jpg)
    若提供既有圖檔清單，則優先比對現有檔案
    """
    if existing_filenames:
        target_name_3d = f"Page-{page_num:03d}.jpg".lower()
        for fn in existing_filenames:
            if Path(fn).name.lower() == target_name_3d:
                return Path(fn).name

        for fn in existing_filenames:
            name = Path(fn).name
            nums = re.findall(r"\d+", name)
            if nums and int(nums[-1]) == page_num:
                return name

    return f"Page-{page_num:03d}.jpg"


def find_default_cover_image(search_dirs: List[Path]) -> Optional[Path]:
    """
    自動搜尋預設封面圖檔 Cover-01.jpg (相容 cover-01.jpg, Cover-01.png, Cover-01.webp 等)
    若無 Cover-01.jpg 則嘗試尋找通用 cover.* 圖檔
    """
    cover01_pattern = re.compile(r"^[Cc]over[-_]?0*1\.(jpg|jpeg|png|webp|tif|tiff)$", re.IGNORECASE)
    fallback_pattern = re.compile(r"^[Cc]over\.(jpg|jpeg|png|webp|tif|tiff)$", re.IGNORECASE)

    # 第一階段：嚴格優先比對 Cover-01.jpg
    for d in search_dirs:
        if not d.exists():
            continue
        if d.is_file():
            if cover01_pattern.match(d.name):
                return d.resolve()
            continue
        try:
            for f in sorted(d.iterdir()):
                if f.is_file() and cover01_pattern.match(f.name):
                    return f.resolve()
        except OSError:
            pass
        for root, _, files in os.walk(d):
            for f in sorted(files):
                if cover01_pattern.match(f):
                    return (Path(root) / f).resolve()

    # 第二階段：比對 cover.jpg
    for d in search_dirs:
        if not d.exists():
            continue
        if d.is_file():
            if fallback_pattern.match(d.name):
                return d.resolve()
            continue
        try:
            for f in sorted(d.iterdir()):
                if f.is_file() and fallback_pattern.match(f.name):
                    return f.resolve()
        except OSError:
            pass
        for root, _, files in os.walk(d):
            for f in sorted(files):
                if fallback_pattern.match(f):
                    return (Path(root) / f).resolve()

    return None


def normalize_layout_direction(val: Optional[str]) -> str:
    """標準化排版方向字串"""
    if not val:
        return "auto"
    s = str(val).strip().lower()
    if s in ("vertical", "vert", "vertical-rl", "v"):
        return "vertical"
    if s in ("horizontal", "horizon", "horizontal-tb", "h"):
        return "horizontal"
    return "auto"


def save_atomic_json(target_path: Union[str, Path], data: Any):
    """
    以原子操作安全寫入 JSON 檔案，先寫至 .tmp 檔再 rename，避免中斷造成損壞檔。
    """
    target_p = Path(target_path)
    target_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = target_p.with_suffix(".json.tmp")
    with open(tmp_p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_p.replace(target_p)


def get_cached_page_result(raw_file: Union[str, Path], output_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    檢查並載入已存在的 raw_pages/{stem}.json 快取。
    若檔案存在、JSON 格式正確、非失敗狀態 (無 status == "failed" 且非 _fail.json)，
    且其中引用的插圖檔案皆實體存在，傳回快取資料；否則傳回 None (以便系統自動重新推論)。
    """
    rf = Path(raw_file)
    if not rf.exists() or rf.stat().st_size == 0:
        return None
    if rf.name.endswith("_fail.json"):
        return None
    try:
        with open(rf, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "blocks" not in data:
            return None
        if data.get("status") == "failed":
            return None
        # 驗證插圖完整性：若引用的插圖檔案遺失，視為快取失效需重跑
        images_dir = Path(output_dir) / "images"
        for b in data.get("blocks", []):
            if isinstance(b, dict) and b.get("type") == "figure":
                img_src = b.get("img_src", "")
                if img_src:
                    fig_name = Path(img_src).name
                    if not (images_dir / fig_name).exists():
                        return None
        return data
    except Exception:
        return None


def render_html(
    page_results: List[Dict[str, Any]],
    title: str = "Scanned Book",
    toc_data: Optional[Dict[str, Any]] = None,
    cover_image: Optional[Union[str, Path]] = None,
    layout_direction: str = "horizontal",
) -> str:
    """
    將所有頁面的結構化識別結果組裝為出版級 HTML5。
    支援目錄導覽、插圖/圖說、表格、MathML 公式與直排/橫排排版。
    """
    body_content = []
    available_page_files = [p["page_id"] for p in page_results]

    # 1. 插入封面區塊 (Cover)
    cover_filename = Path(cover_image).name if cover_image else None
    if cover_filename:
        body_content.append('<section class="book-cover-page" id="book-cover">')
        body_content.append('  <figure class="book-figure fullpage-figure cover-figure">')
        body_content.append(f'    <img src="images/{cover_filename}" alt="封面: {title}" class="cover-image">')
        body_content.append('  </figure>')
        body_content.append('</section>\n<hr class="page-divider">\n')

    # 2. 建立章節對應至起始頁面的映射 (EPUB outline 核心)
    page_chapter_map: Dict[str, List[Dict[str, Any]]] = {}
    if toc_data and "toc" in toc_data and toc_data["toc"]:
        toc_entries = toc_data["toc"]
        page_numbers: List[Tuple[Optional[int], str]] = []
        for p_file in available_page_files:
            nums = re.findall(r"\d+", p_file)
            p_num = int(nums[-1]) if nums else None
            page_numbers.append((p_num, p_file))

        for c_idx, entry in enumerate(toc_entries, start=1):
            start_p_val = entry.get("start_page")
            try:
                start_p = int(start_p_val) if start_p_val is not None and str(start_p_val).strip().lower() != "none" else None
            except (ValueError, TypeError):
                start_p = None

            end_p_val = entry.get("end_page")
            try:
                end_p = int(end_p_val) if end_p_val is not None and str(end_p_val).strip().lower() != "none" else start_p
            except (ValueError, TypeError):
                end_p = start_p

            target_file = None
            if start_p is not None:
                # 優先 1: 精確比對頁碼
                for p_num, p_file in page_numbers:
                    if p_num is not None and p_num == start_p:
                        target_file = p_file
                        break
                # 優先 2: 落在章節頁碼範圍內的第一個頁面
                if not target_file and end_p is not None:
                    for p_num, p_file in page_numbers:
                        if p_num is not None and start_p <= p_num <= end_p:
                            target_file = p_file
                            break
                # 優先 3: 依檔名規則推算 fallback
                if not target_file:
                    target_file = resolve_page_filename(start_p, available_page_files)

            entry_info = dict(entry)
            entry_info["_chapter_index"] = c_idx
            if target_file:
                if target_file not in page_chapter_map:
                    page_chapter_map[target_file] = []
                page_chapter_map[target_file].append(entry_info)

    # 3. 插入目錄索引導覽區塊 (TOC)
    if toc_data and "toc" in toc_data and toc_data["toc"]:
        toc_entries = toc_data["toc"]
        toc_title = toc_data.get("book_title", title)

        body_content.append('<h1 id="toc-heading" class="chapter-heading">目錄</h1>')
        body_content.append('<nav id="toc" class="book-toc" role="doc-toc">')
        body_content.append(f'  <h2 class="toc-title">目錄 ({toc_title})</h2>')
        body_content.append('  <ul class="toc-list">')

        for c_idx, entry in enumerate(toc_entries, start=1):
            start_p_val = entry.get("start_page")
            try:
                start_p = int(start_p_val) if start_p_val is not None and str(start_p_val).strip().lower() != "none" else None
            except (ValueError, TypeError):
                start_p = None

            full_t = entry.get("full_title", entry.get("title", ""))
            target_file = resolve_page_filename(start_p, available_page_files) if start_p is not None else ""

            body_content.append('    <li class="toc-item">')
            if start_p is not None and target_file:
                body_content.append(f'      <a href="#chapter-{c_idx}" class="toc-link">')
                body_content.append(f'        <span class="toc-chapter">{full_t}</span>')
                body_content.append('        <span class="toc-dots"></span>')
                body_content.append(f'        <span class="toc-page">第 {start_p} 頁 ({target_file})</span>')
                body_content.append('      </a>')
            else:
                body_content.append('      <span class="toc-link noclick">')
                body_content.append(f'        <span class="toc-chapter">{full_t}</span>')
                body_content.append('      </span>')
            body_content.append('    </li>')

        body_content.append('  </ul>')
        body_content.append('</nav>\n<hr class="page-divider">\n')

    # 4. 插入各頁內文
    for p_res in page_results:
        page_id = p_res["page_id"]
        if cover_filename and Path(page_id).name.lower() == cover_filename.lower():
            # 封面已於文檔開頭作為獨立封面展示，略過在內文中重複輸出
            continue
        page_stem = p_res.get("page_stem", Path(page_id).stem)

        # 若此頁為章節起始頁，先插入 <h1> 章節標題 (Pandoc EPUB Outline 自動識別)
        if page_id in page_chapter_map:
            for chap in page_chapter_map[page_id]:
                chap_idx = chap.get("_chapter_index", 1)
                full_t = chap.get("full_title", chap.get("title", f"第 {chap_idx} 章"))
                body_content.append(f'<h1 id="chapter-{chap_idx}" class="chapter-heading">{full_t}</h1>\n')

        body_content.append(f'<section class="book-page" id="{page_id}">')
        body_content.append(f'  <a id="{page_stem}" class="page-anchor-target"></a>')
        body_content.append(f'  <div class="page-anchor"><small>{page_id}</small></div>')

        for b in p_res.get("blocks", []):
            b_type = b.get("type")
            if b_type == "title":
                title_text = clean_math_syntax(b.get("content", ""))
                clean_t = re.sub(r"```(?:markdown)?\s*```", "", title_text).strip()
                if clean_t:
                    body_content.append(f'  <h2 class="book-title">{clean_t}</h2>')
            elif b_type == "text":
                content = clean_math_syntax(b.get("content", ""))
                # 檢查正文中是否包含獨立顯示公式塊 ($$ ... $$ 或 \[ ... \])，避免被 \n 切散為多個 <p>
                math_pattern = re.compile(r"(\$\$.*?\$\$|\\\[.*?\\\])", re.DOTALL)
                parts = math_pattern.split(content)
                for part in parts:
                    if not part.strip():
                        continue
                    if (part.startswith("$$") and part.endswith("$$") and len(part) >= 4) or \
                       ((part.startswith(r"\[") or part.startswith("\\[")) and (part.endswith(r"\]") or part.endswith("\\]"))):
                        norm_f = normalize_formula_content(part)
                        if norm_f:
                            body_content.append(f'  <div class="math-block">\n    {norm_f}\n  </div>')
                    else:
                        paragraphs = part.split("\n")
                        for p in paragraphs:
                            p = p.strip()
                            if p:
                                body_content.append(f'  <p>{p}</p>')
            elif b_type == "figure":
                cap = clean_math_syntax(b.get("caption", ""))
                is_full = b.get("is_full_page", False)
                css_class = "book-figure fullpage-figure" if is_full else "book-figure"
                fig_html = f'  <figure class="{css_class}">\n    <img src="{b["img_src"]}" alt="{cap}">\n'
                if cap and not is_full:
                    fig_html += f'    <figcaption>{cap}</figcaption>\n'
                fig_html += '  </figure>'
                body_content.append(fig_html)
            elif b_type == "caption":
                cap_text = clean_math_syntax(b.get("text", ""))
                body_content.append(f'  <p class="book-caption">{cap_text}</p>')
            elif b_type == "table":
                body_content.append(f'  <div class="table-container">\n    {b.get("content", "")}\n  </div>')
            elif b_type == "formula":
                norm_formula = normalize_formula_content(b.get("content", ""))
                if norm_formula:
                    body_content.append(f'  <div class="math-block">\n    {norm_formula}\n  </div>')

        body_content.append('</section>\n<hr class="page-divider">\n')

    body_html_str = "\n".join(body_content)

    body_class = ' class="layout-vertical"' if layout_direction == "vertical" else ''
    vertical_css = """
    body.layout-vertical {
      writing-mode: vertical-rl;
      -webkit-writing-mode: vertical-rl;
      -epub-writing-mode: vertical-rl;
      text-orientation: mixed;
      -webkit-text-orientation: mixed;
      max-width: none;
      height: 90vh;
      overflow-x: auto;
      overflow-y: hidden;
      padding: 30px 40px;
    }
    body.layout-vertical .book-page {
      display: inline-block;
      vertical-align: top;
      height: 85vh;
      margin: 0 1.5em;
      writing-mode: vertical-rl;
      -webkit-writing-mode: vertical-rl;
      -epub-writing-mode: vertical-rl;
    }
    body.layout-vertical p {
      margin: 0 0.8em;
      text-indent: 2em;
      line-height: 2.0;
    }
    body.layout-vertical h1,
    body.layout-vertical h2,
    body.layout-vertical h3 {
      margin: 0 1.2em;
      line-height: 1.4;
    }
    body.layout-vertical .page-anchor {
      text-align: right;
      color: #888;
      font-size: 0.75rem;
      margin: 0 0.5em 0 0;
    }
    body.layout-vertical .book-cover-page {
      display: inline-block;
      vertical-align: top;
      height: 85vh;
      margin: 0 2em;
    }
    body.layout-vertical .book-toc {
      display: inline-block;
      vertical-align: top;
      height: 85vh;
      margin: 0 2em;
      overflow-y: auto;
    }
""" if layout_direction == "vertical" else ""

    full_html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <!-- KaTeX for formula rendering -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
          onload="renderMathInElement(document.body, {{
            delimiters: [
              {{left: '$$', right: '$$', display: true}},
              {{left: '$', right: '$', display: false}},
              {{left: '\\\\[', right: '\\\\]', display: true}},
              {{left: '\\\\(', right: '\\\\)', display: false}}
            ],
            throwOnError: false
          }});"></script>
  <style>
    :root {{
      --primary-color: #2c3e50;
      --bg-color: #fdfdfd;
      --text-color: #2b2b2b;
      --border-color: #e5e5e5;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang TC", "Hiragino Sans GB", "Microsoft JhengHei", "Noto Serif CJK TC", serif;
      background-color: var(--bg-color);
      color: var(--text-color);
      line-height: 1.85;
      max-width: 880px;
      margin: 0 auto;
      padding: 30px 24px;
    }}
    h1, h2, h3, h4 {{
      color: var(--primary-color);
      font-weight: 700;
      margin-top: 1.6em;
      margin-bottom: 0.6em;
      line-height: 1.35;
    }}
    h2.book-title {{
      border-bottom: 2px solid var(--border-color);
      padding-bottom: 0.3em;
    }}
    h1.chapter-heading {{
      font-size: 1.85rem;
      font-weight: 700;
      color: var(--primary-color);
      margin: 44px 0 20px 0;
      padding-bottom: 10px;
      border-bottom: 2px solid var(--accent-color, #3498db);
      page-break-before: always;
      break-before: page;
    }}
    p {{
      margin: 0.9em 0;
      text-align: justify;
      word-break: break-word;
    }}
    .page-anchor {{
      text-align: right;
      color: #888;
      font-size: 0.75rem;
      margin-bottom: 6px;
      letter-spacing: 0.02em;
    }}
    .page-anchor small {{
      font-size: 0.85em;
    }}
    .page-anchor-target {{
      display: block;
      position: relative;
      top: -20px;
      visibility: hidden;
    }}
    .book-toc {{
      margin: 2em 0 3em 0;
      padding: 24px 28px;
      background: #fafbfc;
      border: 1px solid var(--border-color);
      border-radius: 8px;
    }}
    .toc-title {{
      margin-top: 0;
      margin-bottom: 20px;
      text-align: center;
      border-bottom: 2px solid var(--border-color);
      padding-bottom: 12px;
      color: var(--primary-color);
    }}
    .toc-list {{
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    .toc-item {{
      margin: 12px 0;
    }}
    .toc-link {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      text-decoration: none;
      color: var(--text-color);
      font-size: 1.05em;
      transition: color 0.15s ease;
    }}
    .toc-link:hover {{
      color: #2980b9;
    }}
    .toc-chapter {{
      font-weight: 500;
    }}
    .toc-dots {{
      flex: 1;
      border-bottom: 1px dotted #bbb;
      margin: 0 12px;
      min-width: 20px;
    }}
    .toc-page {{
      font-variant-numeric: tabular-nums;
      color: #666;
      font-size: 0.95em;
      white-space: nowrap;
    }}
    .book-cover-page {{
      margin: 0 auto 3em auto;
      text-align: center;
      page-break-after: always;
      break-after: page;
    }}
    .cover-figure {{
      margin: 0 auto;
      max-width: 100%;
    }}
    .cover-image {{
      max-height: 85vh;
      max-width: 100%;
      height: auto;
      box-shadow: 0 8px 24px rgba(0,0,0,0.15);
      border-radius: 4px;
      display: inline-block;
    }}
    .book-figure {{
      margin: 2em auto;
      text-align: center;
    }}
    .book-figure img {{
      max-width: 96%;
      height: auto;
      border: 1px solid var(--border-color);
      box-shadow: 0 4px 12px rgba(0,0,0,0.06);
      border-radius: 6px;
      padding: 4px;
      background: #fff;
    }}
    .fullpage-figure {{
      margin: 1.5em auto;
      text-align: center;
    }}
    .fullpage-figure img {{
      max-width: 100%;
      height: auto;
      border: 1px solid var(--border-color);
      box-shadow: 0 4px 12px rgba(0,0,0,0.06);
      border-radius: 6px;
      display: block;
      margin: 0 auto;
      padding: 0;
      background: #fff;
    }}
    .book-figure figcaption, .book-caption {{
      margin-top: 10px;
      font-size: 0.95em;
      color: #666;
      font-weight: 500;
    }}
    .table-container {{
      margin: 2em 0;
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1em 0;
      font-size: 0.98em;
    }}
    th, td {{
      border: 1px solid #ccc;
      padding: 10px 14px;
      text-align: center;
    }}
    th {{
      background-color: #f7f9fa;
      font-weight: 600;
    }}
    .math-block {{
      margin: 1.6em 0;
      text-align: center;
      overflow-x: auto;
      padding: 8px 0;
    }}
    .page-divider {{
      border: 0;
      height: 1px;
      background: var(--border-color);
      margin: 40px 0;
    }}
    {vertical_css}
  </style>
</head>
<body{body_class}>
{body_html_str}
</body>
</html>
"""
    return sanitize_html_tags(full_html)


def sanitize_epub_math(epub_path: Union[str, Path]) -> int:
    """
    掃描並清理 EPUB 內部 XHTML 檔案中可能殘留的未轉譯 LaTeX 數學語法或 Pandoc 未能解析之 <span class="math ..."> 標記。
    回傳修復的標記數量。
    """
    e_path = Path(epub_path)
    if not e_path.exists():
        return 0

    total_fixes = 0
    temp_epub = e_path.parent / f"__tmp_math_{e_path.name}"

    try:
        with zipfile.ZipFile(e_path, "r") as zin, zipfile.ZipFile(temp_epub, "w", compression=zin.compression) as zout:
            for item in zin.infolist():
                content_bytes = zin.read(item.filename)
                if item.filename.endswith(".xhtml") or item.filename.endswith(".html"):
                    text = content_bytes.decode("utf-8", errors="replace")
                    orig_text = text

                    def fix_math_span(match):
                        inner = match.group(1).strip()
                        if inner.startswith("$") and inner.endswith("$") and len(inner) >= 2:
                            inner = inner[1:-1].strip()
                        elif inner.startswith(r"\(") and inner.endswith(r"\)") and len(inner) >= 4:
                            inner = inner[2:-2].strip()
                        elif inner.startswith(r"\[") and inner.endswith(r"\]") and len(inner) >= 4:
                            inner = inner[2:-2].strip()

                        inner = re.sub(r"\\textcircled\{([^{}]+)\}", lambda m: to_circled_char(m.group(1)), inner)
                        if re.match(r"^" + CIRCLED_CHARS_RE + r"$", inner):
                            return inner
                        m_sup = re.match(r"^\^\{\s*(" + CIRCLED_CHARS_RE + r"|[0-9,，\s\-]+)\s*\}$", inner)
                        if m_sup:
                            return f"<sup>{m_sup.group(1).strip()}</sup>"
                        m_sup2 = re.match(r"^\^(" + CIRCLED_CHARS_RE + r"|[0-9]+)$", inner)
                        if m_sup2:
                            return f"<sup>{m_sup2.group(1)}</sup>"
                        m_word = re.match(r"^([A-Za-z0-9_\u4e00-\u9fa5]*)(?:\\mathrm\{([^{}]+)\})?\s*\^\{\s*(" + CIRCLED_CHARS_RE + r")\s*\}$", inner)
                        if m_word:
                            w = (m_word.group(1) or "") + (m_word.group(2) or "")
                            return f"{w}<sup>{m_word.group(3)}</sup>"
                        return inner

                    text = re.sub(r'<span class="math (?:inline|display)">([\s\S]*?)</span>', fix_math_span, text)

                    if text != orig_text:
                        total_fixes += 1
                        content_bytes = text.encode("utf-8")

                zout.writestr(item, content_bytes)

        if total_fixes > 0:
            temp_epub.replace(e_path)
            print(f"[*] EPUB MathML 校驗修正: 自動替換了 {total_fixes} 處未轉譯之 LaTeX/標籤原始碼")
        else:
            if temp_epub.exists():
                temp_epub.unlink()
    except Exception as e:
        if temp_epub.exists():
            try:
                temp_epub.unlink()
            except Exception:
                pass
        print(f"[-] EPUB 數學標記校驗略過: {e}")

    return total_fixes


def build_epub_with_pandoc(
    html_path: str,
    epub_path: str,
    title: str = "Scanned Book",
    cover_image: Optional[str] = None,
    split_level: int = 1,
    layout_direction: str = "horizontal",
    css_path: Optional[str] = None,
):
    """
    呼叫 Pandoc 編譯 MathML EPUB3 電子書。
    注意：EPUB 一律維持標準橫排排版 (Horizontal-TB / LTR)，直排參數與樣式已全面取消；
    HTML 原生直排設定保持獨立運作不受影響。
    """
    pandoc_bin = "/opt/anaconda3/bin/pandoc"
    if not os.path.exists(pandoc_bin):
        pandoc_bin = "pandoc"

    html_dir = os.path.dirname(os.path.abspath(html_path))
    html_file = os.path.basename(html_path)
    abs_epub_path = os.path.abspath(epub_path)

    # 檢查輸入 HTML 是否包含未轉譯的 \textcircled 或舊版 File: 標記，若有則生成暫存檔預先清洗
    target_compile_file = html_file
    temp_sanitized_html = None
    try:
        with open(html_path, "r", encoding="utf-8", errors="replace") as f_chk:
            raw_html_text = f_chk.read()
        sanitized_html = sanitize_html_tags(clean_math_syntax(raw_html_text))
        cleaned_html, file_prefix_cleaned = re.subn(
            r'(<div\s+class=["\']page-anchor["\'][^>]*>\s*(?:<small>)?)\s*File:\s*',
            r'\1',
            sanitized_html,
            flags=re.IGNORECASE,
        )
        if file_prefix_cleaned > 0:
            sanitized_html = cleaned_html

        if sanitized_html != raw_html_text:
            temp_sanitized_html = Path(html_dir) / f"__epub_temp_{Path(html_file).stem}.html"
            temp_sanitized_html.write_text(sanitized_html, encoding="utf-8")
            target_compile_file = temp_sanitized_html.name
    except Exception as e:
        pass

    # 構造 EPUB 內嵌專屬樣式表 (強制橫排 horizontal-tb)
    auto_css_path = Path(html_dir) / f"__epub_auto_style_{Path(html_file).stem}.css"
    base_epub_css = """/* ScanORC Base EPUB Styles (Strict Horizontal-TB / LTR) */
html, body, body.layout-vertical {
  writing-mode: horizontal-tb !important;
  -webkit-writing-mode: horizontal-tb !important;
  -epub-writing-mode: horizontal-tb !important;
  height: auto !important;
  max-height: none !important;
  overflow: visible !important;
}
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang TC", "Hiragino Sans GB", "Microsoft JhengHei", "Noto Serif CJK TC", serif;
  line-height: 1.85;
  color: #2b2b2b;
}
p {
  margin: 0.8em 0;
  text-align: justify;
}
h1, h2, h3, h4 {
  font-weight: bold;
  line-height: 1.35;
}
h1.chapter-heading {
  font-size: 1.85rem;
  font-weight: 700;
  color: #1a202c;
  margin: 44px 0 20px 0;
  padding-bottom: 10px;
  border-bottom: 2px solid #3498db;
  page-break-before: always;
  break-before: page;
}
.book-page, .book-figure, .book-cover-page, .book-toc {
  writing-mode: horizontal-tb !important;
  -webkit-writing-mode: horizontal-tb !important;
  -epub-writing-mode: horizontal-tb !important;
  height: auto !important;
  display: block !important;
}
.page-anchor {
  text-align: right;
  color: #888;
  font-size: 0.75rem;
  margin-bottom: 6px;
  letter-spacing: 0.02em;
  user-select: none;
}
.page-anchor small {
  font-size: 0.85em;
}
.book-figure {
  margin: 1.5em auto;
  text-align: center;
}
.book-figure img {
  max-width: 96%;
  height: auto;
}
.fullpage-figure img {
  max-width: 100%;
}
.book-caption {
  margin-top: 8px;
  font-size: 0.9em;
  color: #666;
  font-weight: 500;
}
.table-container {
  margin: 1.5em 0;
  overflow-x: auto;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 1em 0;
}
th, td {
  border: 1px solid #ccc;
  padding: 8px 12px;
  text-align: center;
}
th {
  background-color: #f7f9fa;
  font-weight: 600;
}
.math-block {
  margin: 1.5em 0;
  text-align: center;
}
"""

    if css_path and Path(css_path).exists():
        custom_css = Path(css_path).read_text(encoding="utf-8", errors="replace")
        base_epub_css += "\n" + custom_css

    auto_css_path.write_text(base_epub_css, encoding="utf-8")

    cmd = [
        pandoc_bin,
        "-f", "html+tex_math_dollars+tex_math_single_backslash+tex_math_double_backslash",
        target_compile_file,
        "-o", abs_epub_path,
        "--resource-path=.",
        f"--css={auto_css_path.name}",
        "--mathml",
        "--toc",
        "--toc-depth=3",
        f"--split-level={split_level}",
        "--metadata", f"title={title}",
        "--metadata", "language=zh-TW",
    ]
    if cover_image and os.path.exists(cover_image):
        cmd.append(f"--epub-cover-image={os.path.abspath(cover_image)}")

    print(f"[*] 執行 Pandoc 編譯 MathML EPUB (橫排標準排版, 工作目錄: {html_dir}):\n    {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=html_dir, check=True)
        # 執行 EPUB 內部 MathML 與標記二次驗證，確保 100% 無殘留 LaTeX
        sanitize_epub_math(abs_epub_path)
        print(f"[+] EPUB 生成成功 (標準橫排 MathML 渲染與章節 Outline): {abs_epub_path}")
    finally:
        if auto_css_path.exists():
            try:
                auto_css_path.unlink()
            except Exception:
                pass
        if temp_sanitized_html and temp_sanitized_html.exists():
            try:
                temp_sanitized_html.unlink()
            except Exception:
                pass


def raw_pages_to_html(
    raw_dir: Union[str, Path],
    output_html: Optional[Union[str, Path]] = None,
    toc_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    cover_image: Optional[Union[str, Path]] = None,
    layout_direction: str = "auto",
    compile_epub: bool = False,
    cover_first: bool = True,
) -> Path:
    """
    核心 API：讀取 raw_pages 目錄中的 JSON 檔案並組裝生成 HTML (與選擇性 EPUB)。
    """
    raw_path = Path(raw_dir).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"找不到指定的 raw_pages 目錄: {raw_path}")

    # 若傳入的是 output 根目錄，自動探測其底下的 raw_pages
    if raw_path.is_dir() and (raw_path / "raw_pages").is_dir():
        raw_path = raw_path / "raw_pages"

    # 收集 raw_pages 中的所有 JSON 檔案 (排除 .tmp 暫存檔)
    all_json_files = [
        f for f in raw_path.iterdir()
        if f.is_file() and f.suffix.lower() == ".json" and not f.name.endswith(".tmp")
    ]

    if not all_json_files:
        raise ValueError(f"目錄 '{raw_path}' 中未找到任何 OCR raw 檔 (*.json)！")

    # 分類有效正常 JSON 檔與 _fail.json 檔
    normal_map = {}
    fail_map = {}
    for f in all_json_files:
        stem = f.stem
        if stem.endswith("_fail"):
            orig_stem = stem[:-5]
            fail_map[orig_stem] = f
        else:
            normal_map[stem] = f

    # 對於既有 normal 又有 fail 的頁面，以 normal 為準；
    # 若某頁僅有 _fail.json，則載入該備援原圖嵌入以確保書籍完整，並發出醒目警示
    target_json_files = []
    failed_loaded = []
    for stem, f in normal_map.items():
        target_json_files.append(f)

    for stem, f in fail_map.items():
        if stem not in normal_map:
            target_json_files.append(f)
            failed_loaded.append(f.name)

    if failed_loaded:
        print(f"  [!] 警告: 發現 {len(failed_loaded)} 個辨識失敗頁面 (已載入原圖嵌入以確保書籍排版完整): {sorted(failed_loaded)}", file=sys.stderr)

    # 排序 JSON 檔案 (Cover 優先，其次依自然數字排序，以原始頁面 stem 排序)
    sorted_json_files = sorted(
        target_json_files,
        key=lambda f: book_page_sort_key(f.name.replace("_fail.json", ".json"), cover_first=cover_first)
    )

    # 讀取各頁 JSON 內容
    page_results = []
    for jf in sorted_json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "blocks" in data:
                page_results.append(data)
            else:
                print(f"[-] 警告: 檔案 '{jf.name}' 不符合 raw page 格式，已略過", file=sys.stderr)
        except Exception as e:
            print(f"[-] 警告: 讀取 '{jf.name}' 失敗 ({e})，已略過", file=sys.stderr)

    if not page_results:
        raise ValueError(f"目錄 '{raw_path}' 中無有效的 raw page 結構內容！")

    print(f"[*] 成功讀取 {len(page_results)} 個頁面 raw 檔 (來源目錄: {raw_path})")

    # 解析 toc.json
    toc_data = None
    resolved_toc = None
    if toc_path:
        p_toc = Path(toc_path).resolve()
        if p_toc.exists():
            resolved_toc = p_toc
    if not resolved_toc:
        # 自動探尋候選目錄中的 toc.json
        cand_dirs = [raw_path.parent, raw_path, Path.cwd()]
        for d in cand_dirs:
            if (d / "toc.json").exists():
                resolved_toc = (d / "toc.json").resolve()
                break

    if resolved_toc and resolved_toc.exists():
        try:
            with open(resolved_toc, "r", encoding="utf-8") as f:
                toc_data = json.load(f)
            chap_count = len(toc_data.get("toc", []))
            print(f"[*] 成功套用目錄索引: {resolved_toc} (共 {chap_count} 個章節)")
        except Exception as e:
            print(f"[-] 警告: 讀取目錄索引失敗 '{resolved_toc}': {e}", file=sys.stderr)

    # 推導書名 (Title)
    resolved_title = title
    if not resolved_title:
        if toc_data and toc_data.get("book_title"):
            resolved_title = toc_data["book_title"]
        elif raw_path.parent.name and raw_path.parent.name.lower() != "output":
            resolved_title = raw_path.parent.name
        elif raw_path.parent.parent.name:
            resolved_title = raw_path.parent.parent.name
        else:
            resolved_title = "數位化掃描書本"
    print(f"[*] 書名設定: '{resolved_title}'")

    # 決定輸出路徑
    if output_html:
        out_p = Path(output_html).resolve()
        if out_p.is_dir() or out_p.suffix.lower() != ".html":
            out_p.mkdir(parents=True, exist_ok=True)
            out_html_path = out_p / "book.html"
        else:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_html_path = out_p
    else:
        out_html_path = raw_path.parent / "book.html"

    # 解析封面圖檔
    resolved_cover = None
    if cover_image:
        cov_p = Path(cover_image).resolve()
        if cov_p.exists():
            resolved_cover = cov_p
    if not resolved_cover:
        search_dirs = [
            out_html_path.parent / "images",
            raw_path.parent / "images",
            raw_path.parent,
            Path.cwd(),
        ]
        resolved_cover = find_default_cover_image(search_dirs)

    if resolved_cover and resolved_cover.exists():
        print(f"[*] 套用封面圖檔: {resolved_cover}")
        # 確保封面複製至輸出目錄 images/ 下，防止 HTML 引用破圖
        dest_images_dir = out_html_path.parent / "images"
        dest_images_dir.mkdir(parents=True, exist_ok=True)
        dest_cover = dest_images_dir / resolved_cover.name
        if not dest_cover.exists() or dest_cover.resolve() != resolved_cover.resolve():
            try:
                shutil.copy2(resolved_cover, dest_cover)
            except Exception:
                pass

    # 解析排版方向
    eff_direction = normalize_layout_direction(layout_direction)
    if eff_direction == "auto":
        if toc_data and toc_data.get("layout_direction"):
            eff_direction = normalize_layout_direction(toc_data.get("layout_direction"))
        else:
            eff_direction = "horizontal"
    ori_text = "直排 (豎排 / vertical-rl)" if eff_direction == "vertical" else "橫排 (horizontal)"
    print(f"[*] 設定排版方向: {ori_text}")

    # 執行 HTML 渲染
    html_content = render_html(
        page_results=page_results,
        title=resolved_title,
        toc_data=toc_data,
        cover_image=resolved_cover,
        layout_direction=eff_direction,
    )

    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n[+] 完整 HTML 檔案生成完畢 (含 TOC 索引與章節 Outline): {out_html_path}")

    # 選擇性生成 EPUB
    if compile_epub:
        epub_out_path = out_html_path.with_suffix(".epub")
        build_epub_with_pandoc(
            html_path=str(out_html_path),
            epub_path=str(epub_out_path),
            title=resolved_title,
            cover_image=str(resolved_cover) if resolved_cover else None,
            split_level=1,
            layout_direction="horizontal",
        )

    return out_html_path


def main():
    parser = argparse.ArgumentParser(
        description="ScanORC: OCR Raw 檔直接組裝 HTML / EPUB 工具 (無需 GPU / 模型，秒級重新組裝)"
    )
    parser.add_argument(
        "raw_dir",
        nargs="?",
        default=None,
        help="raw_pages 目錄路徑或 output 目錄 (預設: output/raw_pages)"
    )
    parser.add_argument(
        "--input", "-i", "--raw-dir",
        dest="input_dir",
        default=None,
        help="指定 raw_pages 目錄路徑"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="輸出 HTML 檔案路徑 (例如 output/book.html，預設輸出於 raw_pages 上層目錄之 book.html)"
    )
    parser.add_argument(
        "--toc",
        default=None,
        help="目錄索引 JSON 檔案路徑 (例如 toc.json，未指定時自動於 raw_pages 母目錄搜尋)"
    )
    parser.add_argument(
        "--title", "-T",
        default=None,
        help="書名 / 文檔標題 (未指定時優先從 toc.json 自動取得)"
    )
    parser.add_argument(
        "--cover", "--cover-image",
        default=None,
        help="指定封面圖片路徑 (未指定時自動於 images 目錄尋找 Cover-01.jpg)"
    )
    parser.add_argument(
        "--layout-direction", "--direction",
        default="auto",
        choices=["auto", "horizontal", "vertical", "horizon", "vertical-rl", "horizontal-tb"],
        help="排版方向 (預設 auto 優先繼承 toc.json；支援 horizontal 橫排、vertical 直排)"
    )
    parser.add_argument(
        "--vertical",
        dest="layout_direction",
        action="store_const",
        const="vertical",
        default=argparse.SUPPRESS,
        help="快捷選項：強制設定為直排模式 (豎排 / vertical-rl / RTL 翻頁)"
    )
    parser.add_argument(
        "--horizontal",
        dest="layout_direction",
        action="store_const",
        const="horizontal",
        default=argparse.SUPPRESS,
        help="快捷選項：強制設定為橫排模式 (horizontal / LTR 翻頁)"
    )
    parser.add_argument(
        "--epub",
        action="store_true",
        help="生成 HTML 後是否同步調用 Pandoc 編譯為標準 MathML EPUB3 電子書"
    )
    parser.add_argument(
        "--cover-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="頁面編排順序：是否將 Cover-*.json 編排於 Page-*.json 之前 (預設: True)"
    )

    args = attach_preference_and_template_support(parser, "raw_to_html")

    # 決定 raw 目錄
    target_raw_dir = args.input_dir or args.raw_dir
    if not target_raw_dir:
        cand_dirs = [Path("output/raw_pages"), Path("output"), Path("raw_pages")]
        for cd in cand_dirs:
            if cd.exists():
                target_raw_dir = str(cd)
                break
        if not target_raw_dir:
            target_raw_dir = "output/raw_pages"

    try:
        raw_pages_to_html(
            raw_dir=target_raw_dir,
            output_html=args.output,
            toc_path=args.toc,
            title=args.title,
            cover_image=args.cover,
            layout_direction=args.layout_direction,
            compile_epub=args.epub,
            cover_first=args.cover_first,
        )
        print("[🎉] HTML / EPUB 組裝作業順利完成！")
    except Exception as e:
        print(f"\n[-] 錯誤: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
