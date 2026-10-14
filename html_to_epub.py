#!/usr/bin/env python3
"""
ScanORC: HTML 與圖片直接轉 EPUB 工具 (html_to_epub.py)
由 book.html 與 images 目錄直接編譯生成標準 MathML EPUB3 電子書。

特色：
  1. 獨立輕量：無需 GPU 或大型 AI 模型，依賴 Pandoc 於 1~2 秒內秒級編譯完成。
  2. 智慧偵測：自動抽取 HTML 中的 <title> 書名、語系與所有引用圖片；可自動偵測封面圖檔。
  3. 資源校驗：即時檢查 images 目錄中所有插圖的完整性，防止破圖與遺漏。
  4. 數學公式：支援 LaTeX 數學公式自動轉譯為標準 MathML 向量公式 ($...$ 與 $$...$$)。
  5. API 與 CLI 兼備：支援命令列直接執行，亦可作為模組 import convert_html_to_epub。
"""

import os
import re
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

from config_utils import attach_preference_and_template_support

try:
    from recompress_images import recompress_directory
except ImportError:
    recompress_directory = None


def find_pandoc(custom_path: Optional[str] = None) -> Optional[str]:
    """尋找系統中的 Pandoc 執行檔"""
    if custom_path and os.path.exists(custom_path) and os.access(custom_path, os.X_OK):
        return custom_path

    candidates = [
        "/opt/anaconda3/bin/pandoc",
        shutil.which("pandoc"),
        os.path.expanduser("~/.local/bin/pandoc"),
        "/usr/local/bin/pandoc",
        "/usr/bin/pandoc",
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.access(c, os.X_OK):
            return str(c)
    return None


def inspect_html(html_path: Path, images_dir: Optional[Path] = None) -> Dict:
    """
    分析 HTML 內容：提取書名、語系與引用的圖片清單，並檢查圖檔是否存在
    """
    if not html_path.exists():
        raise FileNotFoundError(f"找不到指定的 HTML 檔案: {html_path}")

    content = html_path.read_text(encoding="utf-8", errors="replace")

    # 1. 抽取書名
    title_match = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else html_path.stem

    # 2. 抽取語言
    lang_match = re.search(r'<html[^>]*lang=["\']([^"\']+)["\']', content, re.IGNORECASE)
    lang = lang_match.group(1).strip() if lang_match else "zh-TW"

    # 3. 抽取圖片引用
    img_matches = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', content, re.IGNORECASE)
    unique_imgs = list(dict.fromkeys(img_matches))

    # 4. 驗證圖檔是否存在
    html_dir = html_path.parent
    existing_imgs = []
    missing_imgs = []

    for src in unique_imgs:
        # 嘗試相對 HTML 路徑
        p1 = (html_dir / src).resolve()
        # 嘗試在 images_dir 搜尋純檔名
        p2 = (images_dir / Path(src).name).resolve() if images_dir else None

        if p1.exists():
            existing_imgs.append((src, str(p1)))
        elif p2 and p2.exists():
            existing_imgs.append((src, str(p2)))
        else:
            missing_imgs.append(src)

    return {
        "title": title,
        "lang": lang,
        "total_img_refs": len(img_matches),
        "unique_images": unique_imgs,
        "existing_images": existing_imgs,
        "missing_images": missing_imgs,
    }


def auto_detect_cover(images_dir: Optional[Path], html_dir: Path) -> Optional[Path]:
    """自動偵測封面圖片 (支援 cover.*, Cover-*.jpg, 封面.* 等)"""
    search_dirs = []
    if images_dir and images_dir.exists():
        search_dirs.append(images_dir)
    if html_dir.exists() and html_dir not in search_dirs:
        search_dirs.append(html_dir)

    # 額外擴充搜尋 samples 與 preprocessed 目錄
    for extra in [html_dir.parent / "samples", html_dir / "preprocessed", html_dir.parent / "preprocessed"]:
        if extra.exists() and extra not in search_dirs:
            search_dirs.append(extra)

    cover_patterns = [
        re.compile(r"^cover\.(jpg|jpeg|png|webp)$", re.IGNORECASE),
        re.compile(r"^cover[-_]?\d+\.(jpg|jpeg|png|webp)$", re.IGNORECASE),
        re.compile(r"^封面\.(jpg|jpeg|png|webp)$", re.IGNORECASE),
        re.compile(r"^front[-_]?cover\.(jpg|jpeg|png|webp)$", re.IGNORECASE),
    ]

    for d in search_dirs:
        for f in sorted(d.iterdir()):
            if f.is_file():
                for pat in cover_patterns:
                    if pat.match(f.name):
                        return f.resolve()
    return None


def ensure_epub_outline(
    html_text: str,
    toc_path: Optional[Path] = None,
    html_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[str, bool]:
    """
    確保 HTML 檔案具備頂層 <h1> 章節標題，供 Pandoc EPUB Outline (nav.xhtml / toc.ncx) 識別。
    若發現 HTML 尚未包含章節 <h1>，則嘗試從 toc.json 或內嵌之 <nav id="toc"> 提取章節清單，
    並於各章對應頁面區塊前自動注入 <h1 id="chapter-..." class="chapter-heading">章節標題</h1>。
    """
    # 檢查是否已有足夠的 h1 標題
    h1_matches = re.findall(r'<h1[^>]*class=["\'][^"\']*chapter-heading[^"\']*["\']', html_text, re.IGNORECASE)
    if len(h1_matches) >= 2:
        return html_text, False

    # 1. 取得章節資料
    toc_data = None
    toc_file = toc_path
    if not toc_file and html_dir:
        for c in [html_dir / "toc.json", html_dir.parent / "toc.json", Path("toc.json"), Path("output/toc.json")]:
            if c.exists():
                toc_file = c
                break

    if toc_file and toc_file.exists():
        try:
            with open(toc_file, "r", encoding="utf-8") as f:
                toc_data = json.load(f)
            if verbose:
                print(f"[*] 載入目錄索引檔用於建立 EPUB Outline: {toc_file}")
        except Exception as e:
            if verbose:
                print(f"[-] 讀取目錄索引檔失敗: {e}")

    chapters = []
    if toc_data and "toc" in toc_data:
        for idx, entry in enumerate(toc_data["toc"], start=1):
            chapters.append({
                "index": idx,
                "title": entry.get("full_title", entry.get("title", f"第 {idx} 章")),
                "start_page": entry.get("start_page", 0),
                "end_page": entry.get("end_page", entry.get("start_page", 0)),
            })
    else:
        # 從 <nav id="toc"> 解析
        nav_match = re.search(r'<nav[^>]+id=["\']toc["\'][^>]*>(.*?)</nav>', html_text, re.DOTALL | re.IGNORECASE)
        if nav_match:
            item_matches = re.finditer(
                r'<li[^>]*class=["\']toc-item["\'][^>]*>\s*<a[^>]+href=["\']#([^"\']+)["\'][^>]*>.*?<span[^>]*class=["\']toc-chapter["\'][^>]*>(.*?)</span>',
                nav_match.group(1),
                re.DOTALL | re.IGNORECASE
            )
            for idx, m in enumerate(item_matches, start=1):
                target = m.group(1).strip()
                title_txt = re.sub(r'<[^>]+>', '', m.group(2)).strip()
                nums = re.findall(r'\d+', target)
                p_num = int(nums[-1]) if nums else 0
                chapters.append({
                    "index": idx,
                    "title": title_txt,
                    "start_page": p_num,
                    "end_page": p_num,
                    "target_id": target,
                })

    if not chapters:
        return html_text, False

    # 搜尋 HTML 中的所有頁面 section
    page_sections = list(re.finditer(r'<section[^>]+(?:class=["\'][^"\']*book-page[^"\']*["\'][^>]*id=["\']([^"\']+)["\']|id=["\']([^"\']+)["\'][^>]*class=["\'][^"\']*book-page[^"\']*["\'])', html_text, re.IGNORECASE))
    if not page_sections:
        page_sections = list(re.finditer(r'<section[^>]+id=["\']([^"\']+)["\']', html_text, re.IGNORECASE))

    page_map = {}
    for m in page_sections:
        sec_id = m.group(1) or m.group(2)
        nums = re.findall(r'\d+', sec_id)
        p_num = int(nums[-1]) if nums else None
        page_map[sec_id] = (p_num, m.start())

    # 計算各章節插入點
    insertions = []

    # 檢查是否需要目錄標題
    if not re.search(r'<h1[^>]*id=["\']toc-heading["\']', html_text, re.IGNORECASE):
        nav_pos = html_text.find('<nav id="toc"')
        if nav_pos == -1:
            nav_pos = html_text.find('<nav class="book-toc"')
        if nav_pos != -1:
            insertions.append((nav_pos, '<h1 id="toc-heading" class="chapter-heading">目錄</h1>\n'))

    assigned_targets = set()
    for chap in chapters:
        target_sec_id = None
        if "target_id" in chap and chap["target_id"] in page_map:
            target_sec_id = chap["target_id"]
        if not target_sec_id:
            for s_id, (p_num, _) in page_map.items():
                if p_num is not None and p_num == chap["start_page"]:
                    target_sec_id = s_id
                    break
        if not target_sec_id:
            for s_id, (p_num, _) in page_map.items():
                if p_num is not None and chap["start_page"] <= p_num <= chap["end_page"]:
                    target_sec_id = s_id
                    break

        if target_sec_id and target_sec_id not in assigned_targets:
            assigned_targets.add(target_sec_id)
            pos = page_map[target_sec_id][1]
            h1_html = f'<h1 id="chapter-{chap["index"]}" class="chapter-heading">{chap["title"]}</h1>\n'
            insertions.append((pos, h1_html))

    if not insertions:
        return html_text, False

    # 按位置從後向前插入，避免位移
    insertions.sort(key=lambda x: x[0], reverse=True)
    new_html = html_text
    for pos, h1_html in insertions:
        new_html = new_html[:pos] + h1_html + new_html[pos:]

    # 確保包含 CSS 樣式
    if "h1.chapter-heading" not in new_html:
        css_inject = """
    h1.chapter-heading {
      font-size: 1.85rem;
      font-weight: 700;
      color: var(--primary-color, #1a202c);
      margin: 44px 0 20px 0;
      padding-bottom: 10px;
      border-bottom: 2px solid var(--accent-color, #3498db);
      page-break-before: always;
      break-before: page;
    }
"""
        style_end = new_html.find("</style>")
        if style_end != -1:
            new_html = new_html[:style_end] + css_inject + new_html[style_end:]

    if verbose:
        print(f"[+] 自動為 HTML 注入 {len(insertions)} 個章節標題 (<h1>)，生成完整 EPUB 大綱導覽 (Outline)！")

    return new_html, True


def convert_html_to_epub(
    html_path: str | Path,
    output_path: Optional[str | Path] = None,
    images_dir: Optional[str | Path] = None,
    toc_path: Optional[str | Path] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    lang: Optional[str] = None,
    cover_image: Optional[str | Path] = None,
    css_path: Optional[str | Path] = None,
    toc_depth: int = 3,
    generate_toc: bool = True,
    mathml: bool = True,
    chapter_level: int = 1,
    compress_images: bool = False,
    image_quality: int = 75,
    image_scale: float = 1.0,
    image_max_dim: Optional[int] = None,
    pandoc_bin: Optional[str] = None,
    layout_direction: str = "auto",
    verbose: bool = True,
) -> Path:
    """
    從 HTML 檔案與 images 目錄直接呼叫 Pandoc 生成 EPUB。

    :param html_path: 輸入 HTML 檔案路徑（或包含 book.html 的目錄）
    :param output_path: 輸出 EPUB 檔案路徑 (預設: <html_dir>/<html_stem>.epub)
    :param images_dir: 圖片資源目錄 (預設: 自動定位 HTML 旁的 images/ 目錄)
    :param toc_path: 目錄結構 JSON 檔案路徑 (若未指定則自動在同目錄尋找 toc.json)
    :param title: 書名 (若未指定則自動從 HTML <title> 標籤提取)
    :param author: 作者名稱 (選填)
    :param lang: 語言標籤 (若未指定則自動從 HTML 提取，預設 zh-TW)
    :param cover_image: 封面圖片路徑 (選填，未指定則自動搜尋 cover.* 圖檔)
    :param css_path: 自訂 EPUB 樣式表 CSS 路徑 (選填)
    :param toc_depth: 目錄層級深度 (預設: 3)
    :param generate_toc: 是否生成 EPUB 目錄導覽 (預設: True)
    :param mathml: 是否啟用 MathML 向量數學公式渲染 (預設: True)
    :param chapter_level: EPUB 章節切分標題層級 (預設: 1，即依 <h1> 章節標題切分)
    :param compress_images: 是否於編譯前重新壓縮 images 目錄中的 JPG/PNG 圖檔
    :param image_quality: 圖片壓縮品質 (1-100，預設 75)
    :param image_scale: 圖片尺寸縮放比例 (例如 0.8 或 1.0，預設 1.0)
    :param image_max_dim: 圖片最大長寬限制 (例如 1920，預設 None)
    :param pandoc_bin: 自訂 Pandoc 執行檔路徑 (選填)
    :param verbose: 是否於終端機輸出詳細進度資訊
    :return: 生成的 EPUB 檔案 Path 物件
    """
    # 1. 規範化路徑
    h_path = Path(html_path).resolve()
    if h_path.is_dir():
        candidate1 = h_path / "book.html"
        candidate2 = h_path / "index.html"
        if candidate1.exists():
            h_path = candidate1
        elif candidate2.exists():
            h_path = candidate2
        else:
            raise FileNotFoundError(f"目錄 '{h_path}' 下未找到 book.html 或 index.html！")

    if not h_path.is_file():
        raise FileNotFoundError(f"HTML 檔案不存在: {h_path}")

    html_dir = h_path.parent
    html_name = h_path.name

    # 圖片目錄定位
    if images_dir:
        img_dir = Path(images_dir).resolve()
    else:
        default_img = html_dir / "images"
        img_dir = default_img if default_img.exists() else None

    # 輸出 EPUB 路徑
    if output_path:
        out_epub = Path(output_path).resolve()
        if out_epub.is_dir():
            out_epub = out_epub / f"{h_path.stem}.epub"
    else:
        out_epub = html_dir / f"{h_path.stem}.epub"

    out_epub.parent.mkdir(parents=True, exist_ok=True)

    # 2. 尋找 Pandoc
    pandoc_exec = find_pandoc(pandoc_bin)
    if not pandoc_exec:
        raise RuntimeError(
            "未在系統中找到 Pandoc 執行檔！\n"
            "請透過以下指令安裝 Pandoc：\n"
            "  Ubuntu/Debian: sudo apt update && sudo apt install -y pandoc\n"
            "  Conda: conda install -c conda-forge pandoc"
        )

    # 3. 執行 images 圖檔重新壓縮與縮放 (若啟用)
    if compress_images and img_dir and img_dir.exists() and any(img_dir.iterdir()):
        if recompress_directory is None:
            if verbose:
                print("[-] 警告: 未能載入 recompress_images 模組，略過圖片壓縮步驟")
        else:
            if verbose:
                print(f"[*] 執行 images 目錄圖片重新壓縮 (品質={image_quality}, 縮放={image_scale}, 最大尺寸={image_max_dim})...")
            tmp_opt_dir = img_dir.parent / "images_opt_tmp"
            if tmp_opt_dir.exists():
                shutil.rmtree(tmp_opt_dir)
            try:
                recompress_directory(
                    input_dir=img_dir,
                    output_dir=tmp_opt_dir,
                    qualities=[image_quality],
                    scales=[image_scale],
                    max_dim=image_max_dim,
                    workers=min(os.cpu_count() or 4, 16),
                    verbose=verbose,
                )
                shutil.rmtree(img_dir)
                tmp_opt_dir.rename(img_dir)
                if verbose:
                    print(f"[+] images 目錄圖檔重新壓縮完成！")
            except Exception as e:
                if verbose:
                    print(f"[-] 圖片壓縮過程異常: {e}")
                if tmp_opt_dir.exists():
                    shutil.rmtree(tmp_opt_dir)

    # 4. 分析與檢驗 HTML 內容
    info = inspect_html(h_path, img_dir)
    book_title = title.strip() if title else info["title"]
    book_lang = lang.strip() if lang else info["lang"]

    if verbose:
        print("======================================================================")
        print(" ScanORC: HTML 直接轉 EPUB 工具 (html_to_epub.py)")
        print("======================================================================")
        print(f"[*] 輸入 HTML:    {h_path}")
        print(f"[*] 書籍標題:      {book_title}")
        print(f"[*] 語系設定:      {book_lang}")
        if img_dir and img_dir.exists():
            print(f"[*] 圖片資源目錄:  {img_dir}")
        print(f"[*] 輸出 EPUB:    {out_epub}")
        print(f"[*] Pandoc 路徑:  {pandoc_exec}")

        total_refs = info["total_img_refs"]
        unique_cnt = len(info["unique_images"])
        if total_refs > 0:
            print(f"[*] 偵測到 {total_refs} 次圖片引用 ({unique_cnt} 個獨立圖檔)")
            if info["missing_images"]:
                print(f"[-] ⚠️ 警告：發現 {len(info['missing_images'])} 個圖片引用在目標目錄中未找到：")
                for m in info["missing_images"][:5]:
                    print(f"      - {m}")
                if len(info["missing_images"]) > 5:
                    print(f"      ... 其餘 {len(info['missing_images']) - 5} 個略過顯示")
            else:
                print(f"[+] 圖片校驗通過：所有引用圖檔均完整存在於目錄中")

    # 5. 確保具備 EPUB 大綱目錄 (注入章節 <h1>)
    html_raw = h_path.read_text(encoding="utf-8", errors="replace")
    updated_html, injected = ensure_epub_outline(
        html_raw,
        toc_path=Path(toc_path) if toc_path else None,
        html_dir=html_dir,
        verbose=verbose,
    )

    compile_target_name = html_name
    temp_html_path = None
    if injected:
        temp_html_path = html_dir / f"__epub_temp_{h_path.stem}.html"
        temp_html_path.write_text(updated_html, encoding="utf-8")
        compile_target_name = temp_html_path.name

    # 6. 封面圖片設定
    actual_cover = None
    if cover_image:
        cov_p = Path(cover_image).resolve()
        if cov_p.exists():
            actual_cover = cov_p
        elif verbose:
            print(f"[-] 警告：指定的封面圖檔不存在: {cover_image}")
    else:
        actual_cover = auto_detect_cover(img_dir, html_dir)
        if actual_cover and verbose:
            print(f"[*] 自動偵測並套用封面圖檔: {actual_cover.name}")

    # 7. 組裝 Pandoc 命令
    cmd = [
        pandoc_exec,
        "-f", "html+tex_math_dollars",
        compile_target_name,
        "-o", str(out_epub),
    ]

    res_paths = ["."]
    if img_dir and img_dir.exists():
        res_paths.append(str(img_dir))
        res_paths.append(str(img_dir.parent))
    cmd.append(f"--resource-path={':'.join(res_paths)}")

    if mathml:
        cmd.append("--mathml")

    if generate_toc:
        cmd.append("--toc")
        cmd.append(f"--toc-depth={toc_depth}")

    if chapter_level and chapter_level > 0:
        cmd.append(f"--split-level={chapter_level}")

    cmd.extend(["--metadata", f"title={book_title}"])
    cmd.extend(["--metadata", f"language={book_lang}"])
    if author:
        cmd.extend(["--metadata", f"author={author}"])

    # 判斷是否為直排 (豎排) 排版
    is_vertical = False
    if layout_direction == "vertical":
        is_vertical = True
    elif layout_direction == "auto":
        if "layout-vertical" in html_raw or "vertical-rl" in html_raw:
            is_vertical = True

    if is_vertical:
        cmd.extend([
            "--metadata", "page-progression-direction=rtl",
            "--metadata", "dir=rtl",
        ])
        if verbose:
            print("[*] 啟用 EPUB 直排模式 (RTL 翻頁與 vertical-rl 排版)")

    if actual_cover:
        cmd.append(f"--epub-cover-image={str(actual_cover)}")

    if css_path:
        c_path = Path(css_path).resolve()
        if c_path.exists():
            cmd.append(f"--css={str(c_path)}")
        elif verbose:
            print(f"[-] 警告：指定的 CSS 檔案不存在: {css_path}")

    if verbose:
        print(f"\n[*] 執行 Pandoc 指令 (工作目錄: {html_dir}):")
        print(f"    {' '.join(cmd)}\n")

    try:
        proc = subprocess.run(
            cmd,
            cwd=html_dir,
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            err_msg = proc.stderr.strip() or proc.stdout.strip()
            raise RuntimeError(f"Pandoc 編譯 EPUB 失敗 (結束碼 {proc.returncode}):\n{err_msg}")

        if not out_epub.exists() or out_epub.stat().st_size == 0:
            raise RuntimeError(f"EPUB 生成失敗，未產出有效檔案: {out_epub}")

        file_size_mb = out_epub.stat().st_size / (1024 * 1024)
        if verbose:
            print("======================================================================")
            print(f"[🎉] EPUB 電子書生成成功！(含原生 MathML 與章節大綱 Outline)")
            print(f"      檔案路徑: {out_epub}")
            print(f"      檔案大小: {file_size_mb:.2f} MB ({out_epub.stat().st_size:,} bytes)")
            print("======================================================================")

        return out_epub
    finally:
        if temp_html_path and temp_html_path.exists():
            try:
                temp_html_path.unlink()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="ScanORC: 由 book.html 與 images 目錄直接秒級生成 EPUB3 電子書 (含原生 MathML 渲染與章節 Outline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用範例:
  # 1. 預設轉換 (自動搜尋 output/book.html 或當前目錄之 book.html，自動套用目錄 outline 與封面):
  python html_to_epub.py

  # 2. 指定 HTML 檔案:
  python html_to_epub.py output/book.html

  # 3. 指定輸出路徑、書名與目錄索引:
  python html_to_epub.py output/book.html -o my_book.epub --title "多空操作秘笈" --toc output/toc.json

  # 4. 啟用 images 圖片重新壓縮與縮放:
  python html_to_epub.py output/book.html --compress-images --image-quality 75 --image-scale 0.8
"""
    )

    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="輸入 HTML 檔案路徑或包含 book.html 的目錄 (預設: 自動偵測 output/book.html 或 ./book.html)"
    )
    parser.add_argument(
        "--html", "-i",
        default=None,
        help="指定 HTML 檔案路徑"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="輸出 EPUB 檔案路徑 (預設: 與 HTML 同目錄且同名之 .epub)"
    )
    parser.add_argument(
        "--images", "--images-dir",
        default=None,
        dest="images",
        help="插圖資源目錄 (預設: HTML 所在目錄旁的 images/ 資料夾)"
    )
    parser.add_argument(
        "--toc",
        default=None,
        help="目錄結構 JSON 檔案路徑 (例如 toc.json，未指定時自動尋找同目錄 toc.json)"
    )
    parser.add_argument(
        "--title", "-T",
        default=None,
        help="書籍標題 (預設: 自動從 HTML <title> 標籤中抽取)"
    )
    parser.add_argument(
        "--author", "-a",
        default=None,
        help="書籍作者 (選填)"
    )
    parser.add_argument(
        "--lang", "-l",
        default=None,
        help="語系代碼 (預設: 自動從 HTML 讀取，若無則為 zh-TW)"
    )
    parser.add_argument(
        "--cover", "-C",
        default=None,
        help="封面圖片路徑 (選填，未指定則自動偵測 Cover-*.jpg 或 cover.* 圖檔)"
    )
    parser.add_argument(
        "--compress-images",
        action="store_true",
        help="在編譯 EPUB 前，自動對 images 目錄中的 JPG 與 PNG 圖檔進行重新壓縮與尺寸縮放"
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        default=75,
        help="images 圖檔壓縮品質 (1-100，預設: 75)"
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=1.0,
        help="images 圖檔尺寸縮放比例 (例如 0.8 或 1.0，預設: 1.0)"
    )
    parser.add_argument(
        "--image-max-dim",
        type=int,
        default=None,
        help="images 圖檔最大長寬限制 (例如 1920，預設 None 不限制)"
    )
    parser.add_argument(
        "--css",
        default=None,
        help="自訂樣式表 CSS 檔案路徑 (選填)"
    )
    parser.add_argument(
        "--toc-depth",
        type=int,
        default=3,
        help="目錄深度層級 (預設: 3)"
    )
    parser.add_argument(
        "--no-toc",
        action="store_true",
        help="不生成 EPUB 目錄導覽"
    )
    parser.add_argument(
        "--no-mathml",
        action="store_true",
        help="停用 LaTeX 數學公式至 MathML 向量轉譯"
    )
    parser.add_argument(
        "--chapter-level",
        type=int,
        default=1,
        help="EPUB 章節檔案切分標題層級 (預設: 1，即按 h1 切割)"
    )
    parser.add_argument(
        "--pandoc",
        default=None,
        help="自訂 Pandoc 執行檔路徑"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="精簡輸出，僅顯示必要資訊與錯誤"
    )
    parser.add_argument(
        "--layout-direction",
        choices=["auto", "horizontal", "vertical"],
        default="auto",
        help="排版方向 (auto: 自動從 HTML 特徵判斷, horizontal: 橫排, vertical: 直排 RTL)"
    )

    args = attach_preference_and_template_support(parser, "html_to_epub")

    # 判定 HTML 輸入目標
    target_html = args.html or args.input
    if not target_html:
        candidates = [
            Path("output/book.html"),
            Path("book.html"),
            Path("output/index.html"),
            Path("index.html"),
        ]
        for c in candidates:
            if c.exists():
                target_html = str(c)
                break

    if not target_html:
        print("[-] 錯誤：未指定 HTML 檔案，且未在當前目錄或 output/ 中找到 book.html！", file=sys.stderr)
        print("    使用方式: python html_to_epub.py <path/to/book.html>", file=sys.stderr)
        sys.exit(1)

    try:
        convert_html_to_epub(
            html_path=target_html,
            output_path=args.output,
            images_dir=args.images,
            toc_path=args.toc,
            title=args.title,
            author=args.author,
            lang=args.lang,
            cover_image=args.cover,
            css_path=args.css,
            toc_depth=args.toc_depth,
            generate_toc=not args.no_toc,
            mathml=not args.no_mathml,
            chapter_level=args.chapter_level,
            compress_images=args.compress_images,
            image_quality=args.image_quality,
            image_scale=args.image_scale,
            image_max_dim=args.image_max_dim,
            pandoc_bin=args.pandoc,
            layout_direction=args.layout_direction,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"\n[-] 錯誤: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
