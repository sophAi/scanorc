#!/usr/bin/env python3
"""
ScanORC: 頁面微創替換與快速重新編譯工具 (patch_pages.py)
功能：
  1. 針對已生成的 book.html / book.epub，在人工校對發現特定頁面辨識不良時，
     手動將該頁面替換為原始高解析圖檔 (<figure class="fullpage-figure">)。
  2. 完整保留原頁面的 id 錨點與目錄連結，確保目錄跳轉不失效。
  3. 自動將對應原始圖檔複製至 output/images/。
  4. 呼叫 Pandoc 在 1~2 秒內快速重新編譯生成最新 book.epub（無需耗時重跑全書 OCR）。
  5. 支援自動備份 (.bak) 與一鍵還原 (--revert)。
  6. 支援頁面狀態檢視 (--list)。
"""

import os
import re
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Set, Dict, Optional, Tuple

from config_utils import attach_preference_and_template_support

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def natural_sort_key(s: str) -> List:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def parse_page_targets(pages_args: List[str]) -> Set[str]:
    """
    解析使用者傳入的替換目標清單：
    支援直接傳入檔名 (Page-048.jpg, Page-048)、純數字頁碼 (48)、或清單文字檔 (bad_pages.txt)
    傳回標準化之搜尋鍵集合
    """
    targets = set()
    for item in pages_args:
        item_str = str(item).strip()
        if not item_str:
            continue

        p = Path(item_str)
        if p.is_file() and p.suffix.lower() in (".txt", ".list", ".csv"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            targets.add(line.lower())
                            targets.add(Path(line).stem.lower())
                            nums = re.findall(r"\d+", line)
                            if nums:
                                targets.add(nums[-1])
                                targets.add(str(int(nums[-1])))
                                targets.add(f"page-{int(nums[-1]):03d}.jpg".lower())
                                targets.add(f"page-{int(nums[-1]):03d}".lower())
            except Exception as e:
                print(f"[-] 讀取清單檔案失敗 '{item_str}': {e}", file=sys.stderr)
            continue

        targets.add(item_str.lower())
        targets.add(Path(item_str).stem.lower())
        nums = re.findall(r"\d+", item_str)
        if nums:
            targets.add(nums[-1])
            targets.add(str(int(nums[-1])))
            targets.add(f"page-{int(nums[-1]):03d}.jpg".lower())
            targets.add(f"page-{int(nums[-1]):03d}".lower())

    return targets


def find_original_image(
    page_id: str,
    source_dirs: List[Path]
) -> Optional[Path]:
    """
    在指定來源目錄及其子目錄中搜尋匹配原始圖檔
    """
    target_name = Path(page_id).name.lower()
    target_stem = Path(page_id).stem.lower()
    nums = re.findall(r"\d+", target_stem)
    target_num = int(nums[-1]) if nums else None

    # 1. 優先精確比對完整檔名 (如 Page-048.jpg)
    for s_dir in source_dirs:
        if not s_dir.exists():
            continue
        for root, _, files in os.walk(s_dir):
            for f in files:
                fp = Path(root) / f
                if fp.suffix.lower() in VALID_IMAGE_EXTS and fp.name.lower() == target_name:
                    return fp

    # 2. 次優先比對 stem (如 Page-048)
    for s_dir in source_dirs:
        if not s_dir.exists():
            continue
        for root, _, files in os.walk(s_dir):
            for f in files:
                fp = Path(root) / f
                if fp.suffix.lower() in VALID_IMAGE_EXTS and fp.stem.lower() == target_stem:
                    return fp

    # 3. 備援比對頁碼數字
    if target_num is not None:
        for s_dir in source_dirs:
            if not s_dir.exists():
                continue
            for root, _, files in os.walk(s_dir):
                for f in files:
                    fp = Path(root) / f
                    if fp.suffix.lower() in VALID_IMAGE_EXTS:
                        f_nums = re.findall(r"\d+", fp.stem)
                        if f_nums and int(f_nums[-1]) == target_num:
                            return fp

    return None


def extract_html_title(html_content: str, fallback: str = "Scanned Book") -> str:
    m = re.search(r"<title>(.*?)</title>", html_content, re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return fallback


def build_epub_with_pandoc(html_path: Path, epub_path: Path, title: str):
    try:
        from raw_to_html import build_epub_with_pandoc as _raw_build_epub
        _raw_build_epub(html_path, epub_path, title=title, layout_direction="auto")
        return
    except Exception as e:
        print(f"[*] 嘗試 raw_to_html.build_epub_with_pandoc 失敗 ({e})，使用內建 Pandoc 流程")

    pandoc_bin = "/opt/anaconda3/bin/pandoc"
    if not os.path.exists(pandoc_bin):
        pandoc_bin = shutil.which("pandoc") or "pandoc"

    html_dir = html_path.parent
    html_file = html_path.name
    abs_epub_path = epub_path.resolve()

    # 檢查直排設定
    is_vertical = False
    toc_candidate = html_dir / "toc.json"
    if not toc_candidate.exists():
        toc_candidate = html_dir.parent / "toc.json"
    if toc_candidate.exists():
        try:
            import json
            with open(toc_candidate, "r", encoding="utf-8") as f:
                tdata = json.load(f)
                tdir = str(tdata.get("layout_direction", "")).strip().lower()
                if tdir in ("vertical", "vert", "vertical-rl"):
                    is_vertical = True
        except Exception:
            pass
    if not is_vertical and html_path.exists():
        try:
            h_content = html_path.read_text(encoding="utf-8", errors="replace")
            if "layout-vertical" in h_content or "vertical-rl" in h_content:
                is_vertical = True
        except Exception:
            pass

    auto_css_path = html_dir / f"__epub_auto_style_{html_path.stem}.css"
    base_epub_css = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang TC", "Hiragino Sans GB", "Microsoft JhengHei", "Noto Serif CJK TC", serif;
  line-height: 1.85;
  color: #2b2b2b;
}
.page-anchor {
  text-align: right;
  color: #888;
  font-size: 0.75rem;
  margin-bottom: 6px;
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
"""
    if is_vertical:
        base_epub_css += """
html, body {
  writing-mode: vertical-rl;
  -webkit-writing-mode: vertical-rl;
  -epub-writing-mode: vertical-rl;
  text-orientation: mixed;
  -webkit-text-orientation: mixed;
}
body {
  padding: 20px 30px;
}
p {
  margin: 0 0.8em;
  text-indent: 2em;
  line-height: 2.0;
}
.page-anchor {
  text-align: right;
  color: #888;
  font-size: 0.75rem;
  margin: 0 0.5em 0 0;
}
"""
    auto_css_path.write_text(base_epub_css, encoding="utf-8")

    cmd = [
        pandoc_bin,
        "-f", "html+tex_math_dollars+tex_math_single_backslash+tex_math_double_backslash",
        html_file,
        "-o", str(abs_epub_path),
        "--resource-path=.",
        f"--css={auto_css_path.name}",
        "--mathml",
        "--toc",
        "--toc-depth=3",
        "--metadata", f"title={title}",
        "--metadata", "language=zh-TW",
    ]
    if is_vertical:
        cmd.extend([
            "--metadata", "page-progression-direction=rtl",
            "--metadata", "dir=rtl",
        ])

    print(f"[*] 執行 Pandoc 重新編譯 MathML EPUB (工作目錄: {html_dir}):")
    print(f"    {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=html_dir, check=True)
        print(f"[+] EPUB 生成成功: {abs_epub_path}")
    finally:
        if auto_css_path.exists():
            try:
                auto_css_path.unlink()
            except Exception:
                pass


def list_html_pages(html_path: Path):
    """
    列出 HTML 中的所有頁面及其目前狀態
    """
    if not html_path.exists():
        print(f"[-] 找不到 HTML 檔案: {html_path}", file=sys.stderr)
        return

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    sections = re.findall(
        r'<section\s+class="book-page"\s+id="([^"]+)"[^>]*>(.*?)</section>',
        content,
        re.DOTALL | re.IGNORECASE
    )

    print("\n==================================================")
    print(f" 頁面清單檢視: {html_path.name} (共 {len(sections)} 頁)")
    print("==================================================")
    print(f"{'序號':<5} {'頁面標籤 (id)':<25} {'目前呈現狀態'}")
    print("-" * 55)

    raw_count = 0
    ocr_count = 0

    for idx, (p_id, inner) in enumerate(sections, 1):
        is_raw = "fullpage-figure" in inner or "_raw." in inner
        if is_raw:
            raw_count += 1
            status = "[全頁原圖] 嵌入"
        else:
            ocr_count += 1
            status = "[OCR 辨識] 圖文排版"
        print(f"{idx:<5} {p_id:<25} {status}")

    print("=" * 55)
    print(f"統計: 全書共 {len(sections)} 頁 (OCR 排版: {ocr_count} 頁, 原圖嵌入: {raw_count} 頁)\n")


def main():
    parser = argparse.ArgumentParser(
        description="ScanORC 頁面微創替換與快速重編工具 (patch_pages.py)：將異常頁面無縫替換為原始圖檔"
    )
    parser.add_argument(
        "--pages", "-P",
        nargs="*",
        default=[],
        help="欲替換為原圖的檔案名稱、頁碼或清單文字檔 (例如: --pages Page-048.jpg Page-060.jpg 或 --pages 48 60 或 --pages bad.txt)"
    )
    parser.add_argument(
        "--html",
        default="output/book.html",
        help="待修改的 HTML 檔案路徑 (預設: output/book.html)"
    )
    parser.add_argument(
        "--source", "-s",
        nargs="*",
        default=["output/preprocessed", "preprocessed_samples", "samples"],
        help="原始圖檔搜尋來源目錄 (可指定多個目錄，自動遞迴搜尋子目錄，預設: output/preprocessed, preprocessed_samples, samples)"
    )
    parser.add_argument(
        "--epub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="替換完成後是否重新編譯 EPUB (預設: True)"
    )
    parser.add_argument(
        "--backup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在修改前自動生成 .bak 備份檔案 (預設: True)"
    )
    parser.add_argument(
        "--revert",
        action="store_true",
        help="還原至先前的備份檔案 (book.html.bak)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="檢視 HTML 檔案中所有頁面之狀態清單"
    )
    args = attach_preference_and_template_support(parser, "patch_pages")

    html_path = Path(args.html).resolve()

    # 1. 檢視模式
    if args.list:
        list_html_pages(html_path)
        return

    # 2. 還原模式
    if args.revert:
        bak_path = html_path.with_name(f"{html_path.name}.bak")
        if not bak_path.exists():
            print(f"[-] 錯誤：找不到備份檔案 '{bak_path}'！", file=sys.stderr)
            sys.exit(1)
        shutil.copy2(bak_path, html_path)
        print(f"[+] 成功從備份檔案還原: {html_path}")
        if args.epub:
            with open(html_path, "r", encoding="utf-8") as f:
                title = extract_html_title(f.read())
            epub_path = html_path.with_suffix(".epub")
            build_epub_with_pandoc(html_path, epub_path, title)
        return

    if not args.pages:
        print("[-] 錯誤：未指定任何欲替換的頁面！請使用 --pages 指定檔名或頁碼 (或使用 --help / --list)", file=sys.stderr)
        sys.exit(1)

    if not html_path.exists():
        print(f"[-] 錯誤：找不到目標 HTML 檔案 '{html_path}'！", file=sys.stderr)
        sys.exit(1)

    with open(html_path, "r", encoding="utf-8") as f:
        original_html = f.read()

    book_title = extract_html_title(original_html)
    output_dir = html_path.parent
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 解析目標標記
    target_keys = parse_page_targets(args.pages)
    print("==================================================")
    print(" ScanORC 頁面微創替換與快速重新編譯工具")
    print("==================================================")
    print(f"[*] 目標 HTML: {html_path}")
    print(f"[*] 書籍標題: {book_title}")
    print(f"[*] 待替換標記: {sorted(list(target_keys), key=natural_sort_key)}")

    source_dirs = [Path(d).resolve() for d in args.source]
    print(f"[*] 來源搜尋目錄: {[str(d) for d in source_dirs if d.exists()]}")

    # 搜尋所有 <section class="book-page" id="...">
    section_regex = re.compile(
        r'(<section\s+class="book-page"\s+id="([^"]+)"[^>]*>)(.*?)(</section>)',
        re.DOTALL | re.IGNORECASE
    )

    matches = list(section_regex.finditer(original_html))
    if not matches:
        print("[-] 錯誤：在 HTML 檔案中未找到任何 <section class=\"book-page\"> 頁面標籤！", file=sys.stderr)
        sys.exit(1)

    matched_pages = []
    for m in matches:
        p_id = m.group(2)
        p_stem = Path(p_id).stem
        nums = re.findall(r"\d+", p_stem)
        p_num = nums[-1] if nums else ""
        p_num_int = str(int(p_num)) if p_num else ""

        # 比對是否在目標中
        if (p_id.lower() in target_keys or
            p_stem.lower() in target_keys or
            p_num in target_keys or
            p_num_int in target_keys):
            matched_pages.append((p_id, p_stem, m))

    if not matched_pages:
        print(f"[-] 警告：在 HTML 的 {len(matches)} 個頁面中，未找到任何與 {args.pages} 匹配的頁面！", file=sys.stderr)
        print("    提示：可執行 `python patch_pages.py --list` 查看所有可用的頁面標籤名稱。")
        sys.exit(1)

    print(f"[+] 成功匹配到 {len(matched_pages)} 個頁面準備進行微創替換：")
    for p_id, _, _ in matched_pages:
        print(f"    - {p_id}")

    # 備份原始檔案
    if args.backup:
        bak_path = html_path.with_name(f"{html_path.name}.bak")
        shutil.copy2(html_path, bak_path)
        print(f"[*] 已建立原始檔案備份: {bak_path.name}")

    # 執行逐頁替換
    patched_html = original_html
    replaced_count = 0

    for p_id, p_stem, _ in matched_pages:
        # 尋找原始圖片
        orig_img_path = find_original_image(p_id, source_dirs)
        if not orig_img_path:
            print(f"[-] 錯誤：在所有搜尋目錄中皆找不到頁面 '{p_id}' 對應的原始圖檔！略過此頁", file=sys.stderr)
            continue

        # 複製原圖至 output/images/
        dest_filename = f"{p_stem}_raw{orig_img_path.suffix.lower()}"
        dest_filepath = images_dir / dest_filename
        shutil.copy2(orig_img_path, dest_filepath)
        print(f"  [+] 成功複製原圖: {orig_img_path.name} -> images/{dest_filename}")

        # 構造精準替換正則表達式
        page_pattern = re.compile(
            rf'(<section\s+class="book-page"\s+id="{re.escape(p_id)}"[^>]*>)(.*?)(</section>)',
            re.DOTALL | re.IGNORECASE
        )

        replacement_inner = f"""
  <a id="{p_stem}" class="page-anchor-target"></a>
  <div class="page-anchor"><small>{p_id} (已手動替換為原圖)</small></div>
  <figure class="book-figure fullpage-figure">
    <img src="images/{dest_filename}" alt="{p_id}">
  </figure>
"""

        patched_html = page_pattern.sub(rf'\g<1>{replacement_inner}\g<3>', patched_html)
        replaced_count += 1
        print(f"  [+] 頁面 '{p_id}' 區塊內容已替換為全頁原圖")

    # 儲存更新後的 HTML
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(patched_html)
    print(f"\n[+] 成功更新 HTML 檔案: {html_path} (共修復 {replaced_count} 頁)")

    # 重新編譯 EPUB
    if args.epub and replaced_count > 0:
        epub_path = html_path.with_suffix(".epub")
        print("\n[*] 正在重新編譯 EPUB (耗時約 1~2 秒)...")
        try:
            build_epub_with_pandoc(html_path, epub_path, book_title)
        except Exception as e:
            print(f"[-] 編譯 EPUB 失敗: {e}", file=sys.stderr)

    print("\n==================================================")
    print(f"[🎉] 微創修復完成！共替換 {replaced_count} 個頁面為原始圖檔")
    print(f"     HTML 成果: {html_path}")
    if args.epub:
        print(f"     EPUB 成果: {html_path.with_suffix('.epub')}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
