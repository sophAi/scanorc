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
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple


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
    """自動偵測封面圖片"""
    search_dirs = []
    if images_dir and images_dir.exists():
        search_dirs.append(images_dir)
    if html_dir.exists() and html_dir not in search_dirs:
        search_dirs.append(html_dir)

    cover_patterns = [
        re.compile(r"^cover\.(jpg|jpeg|png|webp)$", re.IGNORECASE),
        re.compile(r"^cover[-_]?0*1\.(jpg|jpeg|png|webp)$", re.IGNORECASE),
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


def convert_html_to_epub(
    html_path: str | Path,
    output_path: Optional[str | Path] = None,
    images_dir: Optional[str | Path] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    lang: Optional[str] = None,
    cover_image: Optional[str | Path] = None,
    css_path: Optional[str | Path] = None,
    toc_depth: int = 3,
    generate_toc: bool = True,
    mathml: bool = True,
    chapter_level: int = 2,
    pandoc_bin: Optional[str] = None,
    verbose: bool = True,
) -> Path:
    """
    從 HTML 檔案與 images 目錄直接呼叫 Pandoc 生成 EPUB。

    :param html_path: 輸入 HTML 檔案路徑（或包含 book.html 的目錄）
    :param output_path: 輸出 EPUB 檔案路徑 (預設: <html_dir>/<html_stem>.epub)
    :param images_dir: 圖片資源目錄 (預設: 自動定位 HTML 旁的 images/ 目錄)
    :param title: 書名 (若未指定則自動從 HTML <title> 標籤提取)
    :param author: 作者名稱 (選填)
    :param lang: 語言標籤 (若未指定則自動從 HTML 提取，預設 zh-TW)
    :param cover_image: 封面圖片路徑 (選填，未指定則自動搜尋 cover.* 圖檔)
    :param css_path: 自訂 EPUB 樣式表 CSS 路徑 (選填)
    :param toc_depth: 目錄層級深度 (預設: 3)
    :param generate_toc: 是否生成 EPUB 目錄導覽 (預設: True)
    :param mathml: 是否啟用 MathML 向量數學公式渲染 (預設: True)
    :param chapter_level: EPUB 章節切分標題層級 (預設: 2，即依 <h2> 切分)
    :param pandoc_bin: 自訂 Pandoc 執行檔路徑 (選填)
    :param verbose: 是否於終端機輸出詳細進度資訊
    :return: 生成的 EPUB 檔案 Path 物件
    """
    # 1. 規範化路徑
    h_path = Path(html_path).resolve()
    if h_path.is_dir():
        # 若傳入為目錄，嘗試尋找 book.html 或 index.html
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
        # 預設同層的 images 目錄
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

    # 3. 分析與檢驗 HTML 內容
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

    # 4. 封面圖片設定
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

    # 5. 組裝 Pandoc 命令
    # 支援公式：html+tex_math_dollars
    cmd = [
        pandoc_exec,
        "-f", "html+tex_math_dollars",
        html_name,
        "-o", str(out_epub),
    ]

    # resource-path: 讓 Pandoc 能夠順利找到相對與絕對圖片
    res_paths = ["."]
    if img_dir and img_dir.exists():
        res_paths.append(str(img_dir))
        res_paths.append(str(img_dir.parent))
    cmd.append(f"--resource-path={':'.join(res_paths)}")

    # 數學公式設定
    if mathml:
        cmd.append("--mathml")

    # 目錄設定
    if generate_toc:
        cmd.append("--toc")
        cmd.append(f"--toc-depth={toc_depth}")

    # 章節切割層級 (通常為 2，按 <h2> 分割 XHTML 檔)
    if chapter_level and chapter_level > 0:
        cmd.append(f"--epub-chapter-level={chapter_level}")

    # 元資料
    cmd.extend(["--metadata", f"title={book_title}"])
    cmd.extend(["--metadata", f"language={book_lang}"])
    if author:
        cmd.extend(["--metadata", f"author={author}"])

    # 封面
    if actual_cover:
        cmd.append(f"--epub-cover-image={str(actual_cover)}")

    # 自訂 CSS
    if css_path:
        c_path = Path(css_path).resolve()
        if c_path.exists():
            cmd.append(f"--css={str(c_path)}")
        elif verbose:
            print(f"[-] 警告：指定的 CSS 檔案不存在: {css_path}")

    if verbose:
        print(f"\n[*] 執行 Pandoc 指令 (工作目錄: {html_dir}):")
        print(f"    {' '.join(cmd)}\n")

    # 6. 執行 Pandoc
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
        print(f"[🎉] EPUB 電子書生成成功！")
        print(f"      檔案路徑: {out_epub}")
        print(f"      檔案大小: {file_size_mb:.2f} MB ({out_epub.stat().st_size:,} bytes)")
        print("======================================================================")

    return out_epub


def main():
    parser = argparse.ArgumentParser(
        description="ScanORC: 由 book.html 與 images 目錄直接秒級生成 EPUB3 電子書 (含原生 MathML 渲染)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用範例:
  # 1. 預設轉換 (自動搜尋 output/book.html 或當前目錄之 book.html):
  python html_to_epub.py

  # 2. 指定 HTML 檔案:
  python html_to_epub.py output/book.html

  # 3. 指定輸出路徑與書名:
  python html_to_epub.py output/book.html -o my_book.epub --title "量子力學基礎"

  # 4. 指定圖片目錄與封面:
  python html_to_epub.py output/book.html --images output/images --cover output/images/cover.jpg
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
        "--title", "-t",
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
        "--cover", "-c",
        default=None,
        help="封面圖片路徑 (選填，未指定則自動偵測 cover.* 圖檔)"
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
        default=2,
        help="EPUB 章節檔案切分標題層級 (預設: 2，即按 h2 切割)"
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

    args = parser.parse_args()

    # 判定 HTML 輸入目標
    target_html = args.html or args.input
    if not target_html:
        # 自動搜尋候選路徑
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
            title=args.title,
            author=args.author,
            lang=args.lang,
            cover_image=args.cover,
            css_path=args.css,
            toc_depth=args.toc_depth,
            generate_toc=not args.no_toc,
            mathml=not args.no_mathml,
            chapter_level=args.chapter_level,
            pandoc_bin=args.pandoc,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"\n[-] 錯誤: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
