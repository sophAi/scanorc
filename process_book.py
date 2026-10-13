#!/usr/bin/env python3
"""
ScanORC: 掃描書本全流程一鍵端到端處理腳本 (process_book.py)
整合現有工具：
  [步驟 1/4] 歪斜轉正與邊緣裁切       -> preprocess_scantailor.py (專案自帶 ScanTailor-CLI)
  [步驟 2/4] 跨頁目錄識別與抽取       -> extract_toc.py (GLM-OCR -> toc.json)
  [步驟 3/4] 語意 HTML 與 EPUB 生成  -> pipeline.py (DocLayout-YOLO + GLM-OCR)
  [步驟 4/4] 章節目錄自動建立與歸檔   -> extract_toc.py (--organize-pages)
"""

import os
import re
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def natural_sort_key(p: Path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", p.name)]


def find_cover_images(directory: Path) -> List[Path]:
    """
    在目錄中搜尋所有目錄/封面圖檔 (Cover-*.jpg 或包含 cover/封面 的圖檔)
    """
    covers = []
    for root, _, files in os.walk(directory):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in VALID_IMAGE_EXTS:
                stem_lower = p.stem.lower()
                if re.search(r"(^cover|^封面|^front[-_]?cover)", stem_lower):
                    covers.append(p)
    covers.sort(key=natural_sort_key)
    return covers


def print_banner(step_idx: int, total_steps: int, title: str):
    print("\n" + "=" * 70)
    print(f" [步驟 {step_idx}/{total_steps}] {title}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="ScanORC 端到端一鍵書籍數位化流水線：轉正裁切 -> 目錄抽取 -> EPUB/HTML 生成 -> 章節自動歸檔"
    )
    parser.add_argument(
        "--input", "-i",
        default="samples",
        help="輸入書籍原始圖檔目錄 (包含 Cover-xx.jpg 與 Page-xxx.jpg，預設: samples)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="輸出成果目錄 (預設: output)"
    )
    parser.add_argument(
        "--title", "-t",
        default="",
        help="書名 / 文檔標題 (若未指定則自動從輸入目錄名稱或 toc.json 讀取)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="輸出解析度 DPI (預設 None 保持 1:1 原生尺寸無損輸出，僅指定時縮放)"
    )
    parser.add_argument(
        "--margins",
        type=int,
        default=10,
        help="版心四周均勻邊距大小 (單位: mm，預設: 10)"
    )
    parser.add_argument(
        "--color-mode",
        choices=["color_grayscale", "mixed", "black_and_white"],
        default="color_grayscale",
        help="轉正裁切色彩模式 (預設: color_grayscale 保留插圖階調)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="單頁 OCR 區塊批次大小 (預設 2，16GB 顯卡可設 4)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="多進程並行 Worker 數 (預設 1，16GB 顯卡可設 3~4)"
    )
    parser.add_argument(
        "--skip-ocr", "--skip-orc", "--no-ocr",
        nargs="*",
        default=[],
        dest="skip_ocr",
        help="略過 OCR / ScanTailor 處理的檔案清單 (可指定檔名、頁碼或包含清單的純文字檔路徑，直接以原始圖檔導出並嵌入 HTML/EPUB)"
    )
    parser.add_argument(
        "--epub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否生成 MathML EPUB3 電子書 (預設: True)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="章節歸檔時改為複製圖檔而非移動圖檔 (預設: 移動圖檔)"
    )
    parser.add_argument(
        "--organize-target",
        choices=["preprocessed", "input"],
        default="preprocessed",
        help="執行章節資料夾建立與 Page-xxx.jpg 歸檔之目標 (預設: preprocessed 轉正後圖檔目錄)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.exists():
        print(f"[-] 錯誤：找不到輸入目錄 '{input_dir}'！", file=sys.stderr)
        sys.exit(1)

    book_title = args.title or input_dir.name or "數位化掃描書籍"
    preprocessed_dir = output_dir / "preprocessed"
    toc_json_path = output_dir / "toc.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    print("######################################################################")
    print(" ScanORC 書籍全流程自動化建置系統 (End-to-End Book Digitizer)")
    print("######################################################################")
    print(f"[*] 書籍標題: {book_title}")
    print(f"[*] 輸入目錄: {input_dir}")
    print(f"[*] 輸出目錄: {output_dir}")
    print(f"[*] 轉正目錄: {preprocessed_dir}")
    if args.skip_ocr:
        print(f"[*] 免 OCR 頁面清單: {args.skip_ocr}")
    print("######################################################################")

    total_start_time = time.time()
    TOTAL_STEPS = 4

    # -------------------------------------------------------------------------
    # [步驟 1/4] 歪斜轉正與邊緣裁切 (preprocess_scantailor.py)
    # -------------------------------------------------------------------------
    print_banner(1, TOTAL_STEPS, "影像前置處理：歪斜轉正、黑邊去除與版心標準化")
    step1_cmd = [
        sys.executable, str(script_dir / "preprocess_scantailor.py"),
        "--input", str(input_dir),
        "--output", str(preprocessed_dir),
        "--margins", str(args.margins),
        "--color-mode", args.color_mode,
    ]
    if args.dpi:
        step1_cmd.extend(["--dpi", str(args.dpi)])
    if args.skip_ocr:
        step1_cmd.extend(["--skip-orc", *args.skip_ocr])

    print(f"[*] 執行指令: {' '.join(step1_cmd)}")
    subprocess.run(step1_cmd, check=True)

    # -------------------------------------------------------------------------
    # [步驟 2/4] 跨頁目錄識別與抽取 (extract_toc.py)
    # -------------------------------------------------------------------------
    print_banner(2, TOTAL_STEPS, "目錄頁次結構化抽取 (GLM-OCR -> toc.json)")
    # 優先從轉正後的目錄中尋找 Cover-*.jpg 圖檔
    cover_images = find_cover_images(preprocessed_dir)
    if not cover_images:
        # 若轉正目錄無 cover，嘗試在原始目錄尋找
        cover_images = find_cover_images(input_dir)

    has_toc = False
    if cover_images:
        print(f"[*] 發現 {len(cover_images)} 個目錄頁圖檔：{[p.name for p in cover_images]}")
        step2_cmd = [
            sys.executable, str(script_dir / "extract_toc.py"),
            "--input", *[str(p) for p in cover_images],
            "--output", str(toc_json_path),
            "--book-title", book_title,
        ]
        print(f"[*] 執行指令: {' '.join(step2_cmd)}")
        try:
            subprocess.run(step2_cmd, check=True)
            has_toc = toc_json_path.exists()
            if has_toc:
                print(f"[+] 成功生成目錄結構索引 JSON: {toc_json_path}")
        except subprocess.CalledProcessError as e:
            print(f"[-] 警告：目錄結構抽取過程異常 ({e})，後續將以無目錄索引模式繼續執行", file=sys.stderr)
    else:
        print("[-] 未在輸入目錄中發現 Cover-*.jpg 目錄頁圖檔，跳過目錄抽取步驟")

    # -------------------------------------------------------------------------
    # [步驟 3/4] 語意 HTML 與 EPUB 生成 (pipeline.py)
    # -------------------------------------------------------------------------
    print_banner(3, TOTAL_STEPS, "AI 版面分析、OCR 與 HTML/EPUB 電子書生成")
    step3_cmd = [
        sys.executable, str(script_dir / "pipeline.py"),
        "--samples", str(preprocessed_dir),
        "--output", str(output_dir),
        "--title", book_title,
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
    ]
    if has_toc:
        step3_cmd.extend(["--toc", str(toc_json_path)])
    if args.epub:
        step3_cmd.append("--epub")
    if args.skip_ocr:
        step3_cmd.extend(["--skip-ocr", *args.skip_ocr])

    print(f"[*] 執行指令: {' '.join(step3_cmd)}")
    subprocess.run(step3_cmd, check=True)

    # -------------------------------------------------------------------------
    # [步驟 4/4] 根據 toc.json 生成章節目錄並將 Page-xxx.jpg 歸檔 (extract_toc.py --organize-pages)
    # -------------------------------------------------------------------------
    print_banner(4, TOTAL_STEPS, "章節目錄結構自動建立與圖檔歸檔")
    if has_toc and cover_images:
        target_dir = preprocessed_dir if args.organize_target == "preprocessed" else input_dir
        print(f"[*] 準備對目錄進行章節歸檔: {target_dir}")
        step4_cmd = [
            sys.executable, str(script_dir / "extract_toc.py"),
            "--input", *[str(p) for p in cover_images],
            "--output", str(toc_json_path),
            "--organize-pages", str(target_dir),
        ]
        if args.copy:
            step4_cmd.append("--copy")

        print(f"[*] 執行指令: {' '.join(step4_cmd)}")
        subprocess.run(step4_cmd, check=True)
        print(f"[+] 章節資料夾建立與圖檔歸檔完成！")
    else:
        print("[-] 由於缺少 toc.json 或目錄資料，略過章節歸檔步驟")

    # -------------------------------------------------------------------------
    # 完成總結
    # -------------------------------------------------------------------------
    elapsed = time.time() - total_start_time
    print("\n" + "#" * 70)
    print(" [🎉] ScanORC 書籍一鍵全流程建置成功完成！")
    print(f"      總執行耗時: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分鐘)")
    print(f"      HTML 成果: {output_dir / 'book.html'}")
    if args.epub and (output_dir / 'book.epub').exists():
        print(f"      EPUB 成果: {output_dir / 'book.epub'}")
    if has_toc:
        print(f"      目錄索引: {toc_json_path}")
    print(f"      歸檔目錄: {preprocessed_dir if args.organize_target == 'preprocessed' else input_dir}")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
