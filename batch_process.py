#!/usr/bin/env python3
"""
ScanORC: 多書籍目錄批次數位化調度器 (batch_process.py)
功能：
  1. 支援母目錄自動掃描 (--input-dir) 或多目錄清單 (--dirs)，以書為單位進行循序佇列處理。
  2. 每個書籍目錄支援專屬設定檔 (book_config.json 或自訂名稱) 與現成 toc.json 自動優先沿用。
  3. 支援單一 JSON 設定直通 (python batch_process.py batch_config.json)、-p (--preference) 與 -t (--template)。
  4. 支援中斷續傳 (--skip-existing)、排程預覽 (--dry-run) 與失敗隔離 (--continue-on-error)。
  5. 批次完成後自動產出結構化執行報告 (batch_report.json 與 batch_report.md)。
"""

import os
import re
import sys
import time
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from config_utils import attach_preference_and_template_support, load_json_with_comments

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def natural_sort_key(s: Any) -> List:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def has_images(directory: Path) -> bool:
    """檢查目錄 (或其子目錄) 中是否含有支援的圖檔"""
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in VALID_IMAGE_EXTS:
                return True
    return False


def find_book_config(directory: Path, custom_name: Optional[str] = None) -> Optional[Path]:
    """在書籍目錄中搜尋設定檔"""
    candidates = []
    if custom_name:
        candidates.append(directory / custom_name)
    candidates.extend([
        directory / "book_config.json",
        directory / "preference.json",
        directory / "process_book.json",
        directory / "scanorc.json"
    ])
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def main():
    parser = argparse.ArgumentParser(
        description="ScanORC Batch Processor: 多書籍目錄批次排程與自動化建置工具"
    )

    # 1. 輸入與目錄發現
    parser.add_argument(
        "--input-dir", "-i",
        default=None,
        help="母目錄路徑：自動掃描其第一層子目錄作為各書籍來源 (例如: library/)"
    )
    parser.add_argument(
        "--dirs", "-d",
        nargs="*",
        default=[],
        help="明確指定一個或多個書籍圖檔目錄 (例如: -d /path/book1 /path/book2)"
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="子目錄過濾規則 (搭配 --input-dir 使用，例如: 'Vol-*' 或 'Math_*'，預設: '*')"
    )
    parser.add_argument(
        "--config-name",
        default="book_config.json",
        help="各書籍目錄內偏好設定檔檔名 (預設: book_config.json)"
    )

    # 2. 輸出設定
    parser.add_argument(
        "--output-root", "-o",
        default="output_batch",
        help="批次成果總輸出目錄 (每本書之成果存放於 <output-root>/<書名>/，預設: output_batch)"
    )
    parser.add_argument(
        "--report",
        default=None,
        help="批次彙整報告輸出路徑 (預設: <output-root>/batch_report.json)"
    )

    # 3. 執行控制與容錯
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="斷點續傳：若該書輸出目錄已存在 book.epub (或 book.html) 則自動略過"
    )
    parser.add_argument(
        "--use-existing-toc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="若書籍目錄中存在 toc.json，是否優先沿用並跳過 AI 抽取 (預設: True)"
    )
    parser.add_argument(
        "--force-toc",
        action="store_true",
        help="強制忽略既有 toc.json，一律重新以 AI 抽取目錄"
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="單本書籍處理失敗時記錄錯誤並繼續處理下一本 (預設: True)"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="任一本書籍處理失敗時立即中斷整個批次佇列"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="僅解析掃描各目錄設定與 TOC 狀態並輸出預覽清單，不實際執行運算"
    )

    # 4. 全域預設參數 (傳遞至每本書之 process_book.py)
    parser.add_argument("--title", "-T", default="", help="預設書名標題 (若留空則使用目錄名或 toc.json)")
    parser.add_argument("--cover", default=None, help="預設封面圖片路徑")
    parser.add_argument("--dpi", type=int, default=None, help="預設解析度 DPI (預設 None 保持 1:1)")
    parser.add_argument("--margins", type=int, default=1, help="預設版心四周邊距大小 (mm，預設: 1)")
    parser.add_argument("--quality", type=int, default=85, help="預設 JPG 壓縮品質 (預設: 85)")
    parser.add_argument(
        "--color-mode",
        choices=["color_grayscale", "mixed", "black_and_white"],
        default="color_grayscale",
        help="預設色彩模式 (預設: color_grayscale)"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="單頁 OCR 批次大小 (預設: 2)")
    parser.add_argument("--workers", type=int, default=1, help="單書並行 Worker 數 (預設: 1)")
    parser.add_argument(
        "--skip-ocr",
        nargs="*",
        default=["Cover-*.jpg", "Content-*.jpg"],
        help="預設免 OCR 頁面清單 (預設: ['Cover-*.jpg', 'Content-*.jpg'])"
    )
    parser.add_argument(
        "--epub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否生成 MathML EPUB3 電子書 (預設: True)"
    )
    parser.add_argument(
        "--compress-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否壓縮與縮放 images 插圖 (預設: True)"
    )
    parser.add_argument("--image-quality", type=int, default=75, help="images 圖片品質 (預設: 75)")
    parser.add_argument("--image-scale", type=float, default=0.75, help="images 圖片縮放 (預設: 0.75)")
    parser.add_argument("--image-max-dim", type=int, default=None, help="images 圖片長寬限制 (預設: None)")
    parser.add_argument(
        "--organize-pages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否依目錄章節自動歸檔圖檔 (預設: True)"
    )
    parser.add_argument("--copy", action="store_true", help="章節歸檔使用複製而非移動")
    parser.add_argument(
        "--organize-target",
        choices=["preprocessed", "input"],
        default="preprocessed",
        help="章節歸檔目標目錄 (預設: preprocessed)"
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="AI 模型權重母目錄 (若指定則傳遞至每本書之 process_book.py；亦可透過 SCANORC_MODELS_DIR 設定)"
    )

    args = attach_preference_and_template_support(parser, "batch_process")

    script_dir = Path(__file__).parent.resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. 蒐集所有待處理的書籍目錄
    book_dirs = []
    if args.input_dir:
        root_in = Path(args.input_dir).resolve()
        if not root_in.is_dir():
            print(f"[-] 錯誤：找不到指定的母目錄 '{root_in}'！", file=sys.stderr)
            sys.exit(1)
        for child in root_in.glob(args.pattern):
            if child.is_dir() and has_images(child):
                book_dirs.append(child)

    if args.dirs:
        for d in args.dirs:
            dp = Path(d).resolve()
            if dp.is_dir() and dp not in book_dirs:
                book_dirs.append(dp)

    if not book_dirs and Path("samples").is_dir():
        book_dirs.append(Path("samples").resolve())

    book_dirs.sort(key=natural_sort_key)

    if not book_dirs:
        print("[-] 未在指定來源中找到任何包含圖檔的書籍目錄！", file=sys.stderr)
        print("    請使用 --input-dir 指定母目錄，或使用 --dirs 指定單一或多個目錄。", file=sys.stderr)
        sys.exit(1)

    print("======================================================================")
    print(" ScanORC 多書籍目錄批次數位化排程系統 (Batch Book Digitizer)")
    print("======================================================================")
    print(f"[*] 發現書籍目錄數量: {len(book_dirs)} 本")
    print(f"[*] 成果輸出總目錄: {output_root}")
    print(f"[*] 智慧沿用 TOC: {'啟用' if args.use_existing_toc and not args.force_toc else '關閉'}")
    print(f"[*] 中斷續傳略過已完成: {'啟用' if args.skip_existing else '關閉'}")
    print("======================================================================\n")

    # 2. 構建每本書的執行配置
    book_tasks = []
    for idx, b_dir in enumerate(book_dirs, 1):
        # 尋找書本專屬設定
        cfg_file = find_book_config(b_dir, args.config_name)
        cfg_data = {}
        if cfg_file:
            try:
                cfg_data = load_json_with_comments(cfg_file)
                if not isinstance(cfg_data, dict):
                    cfg_data = {}
            except Exception as e:
                print(f"[!] 警告：讀取書籍設定檔 '{cfg_file.name}' 失敗: {e}，將使用預設設定")

        # 優先決定書名
        title = cfg_data.get("title") or args.title or b_dir.name
        # 輸出子目錄
        out_sub_dir = output_root / b_dir.name

        # 檢查既有 TOC
        custom_toc = cfg_data.get("toc") or cfg_data.get("toc_file")
        has_existing_toc = False
        toc_path = None
        if not args.force_toc and args.use_existing_toc:
            if custom_toc and (b_dir / custom_toc).exists():
                toc_path = (b_dir / custom_toc).resolve()
                has_existing_toc = True
            elif (b_dir / "toc.json").exists():
                toc_path = (b_dir / "toc.json").resolve()
                has_existing_toc = True
            elif (out_sub_dir / "toc.json").exists():
                toc_path = (out_sub_dir / "toc.json").resolve()
                has_existing_toc = True

        # 檢查是否已完成 (續傳)
        is_completed = False
        check_target = out_sub_dir / ("book.epub" if args.epub else "book.html")
        if check_target.is_file() and check_target.stat().st_size > 0:
            is_completed = True

        book_tasks.append({
            "index": idx,
            "book_dir": b_dir,
            "title": title,
            "output_dir": out_sub_dir,
            "config_file": cfg_file,
            "config_data": cfg_data,
            "toc_path": toc_path,
            "has_existing_toc": has_existing_toc,
            "is_completed": is_completed,
        })

    # 3. Dry-Run 預覽模式
    if args.dry_run:
        print(" [🔍] 批次執行排程預覽 (Dry Run)：\n")
        print(f"{'編號':<5} {'書籍目錄名稱':<25} {'書名標題':<25} {'TOC 狀態':<18} {'排程狀態'}")
        print("-" * 85)
        for t in book_tasks:
            toc_status = "現成沿用" if t["has_existing_toc"] else "AI 抽取"
            run_status = "略過 (已完成)" if (t["is_completed"] and args.skip_existing) else "待處理"
            print(f"{t['index']:<5} {t['book_dir'].name:<25} {t['title']:<25} {toc_status:<18} {run_status}")
        print("-" * 85)
        pending_cnt = sum(1 for t in book_tasks if not (t["is_completed"] and args.skip_existing))
        skip_cnt = len(book_tasks) - pending_cnt
        print(f"總計: {len(book_tasks)} 本 | 待執行: {pending_cnt} 本 | 續傳略過: {skip_cnt} 本\n")
        return

    # 4. 正式循序執行佇列
    batch_start_time = time.time()
    results = []

    for t in book_tasks:
        idx = t["index"]
        b_dir = t["book_dir"]
        out_sub_dir = t["output_dir"]
        title = t["title"]
        cfg_data = t["config_data"]

        print("\n" + "=" * 70)
        print(f" >>> [書籍 {idx}/{len(book_tasks)}] {title} ({b_dir.name})")
        print("=" * 70)

        # 檢查斷點續傳
        if t["is_completed"] and args.skip_existing:
            print(f"[+] 檢測到成果已存在: {out_sub_dir / ('book.epub' if args.epub else 'book.html')}")
            print(f"    依 --skip-existing 設定略過此書。")
            results.append({
                "index": idx,
                "title": title,
                "directory": str(b_dir),
                "output_dir": str(out_sub_dir),
                "status": "SKIPPED",
                "elapsed": 0.0,
                "error": None
            })
            continue

        # 組合 process_book.py 指令
        cmd = [
            sys.executable, str(script_dir / "process_book.py"),
            "--input", str(b_dir),
            "--output", str(out_sub_dir),
            "--title", title,
        ]

        # 優先傳遞專屬設定，其次為全域設定
        workers = cfg_data.get("workers", args.workers)
        batch_size = cfg_data.get("batch_size", args.batch_size)
        margins = cfg_data.get("margins", args.margins)
        quality = cfg_data.get("quality", args.quality)
        color_mode = cfg_data.get("color_mode", args.color_mode)
        dpi = cfg_data.get("dpi", args.dpi)
        cover = cfg_data.get("cover", args.cover)

        cmd.extend(["--workers", str(workers)])
        cmd.extend(["--batch-size", str(batch_size)])
        cmd.extend(["--margins", str(margins)])
        cmd.extend(["--quality", str(quality)])
        cmd.extend(["--color-mode", str(color_mode)])
        if dpi:
            cmd.extend(["--dpi", str(dpi)])
        if cover:
            cmd.extend(["--cover", str(cover)])
        models_dir = cfg_data.get("models_dir", args.models_dir)
        if models_dir:
            cmd.extend(["--models-dir", str(models_dir)])

        # TOC
        if t["has_existing_toc"] and t["toc_path"]:
            cmd.extend(["--toc", str(t["toc_path"])])
        if args.force_toc or cfg_data.get("force_toc"):
            cmd.append("--force-toc")

        # 免 OCR 頁面
        skip_ocr = cfg_data.get("skip_ocr") if "skip_ocr" in cfg_data else args.skip_ocr
        if skip_ocr:
            if isinstance(skip_ocr, list):
                cmd.extend(["--skip-ocr", *[str(s) for s in skip_ocr]])
            else:
                cmd.extend(["--skip-ocr", str(skip_ocr)])

        # EPUB
        epub_flag = cfg_data.get("epub", args.epub)
        if not epub_flag:
            cmd.append("--no-epub")

        # 圖片優化
        compress_images = cfg_data.get("compress_images", args.compress_images)
        if compress_images:
            cmd.append("--compress-images")
            img_q = cfg_data.get("image_quality", args.image_quality)
            img_s = cfg_data.get("image_scale", args.image_scale)
            cmd.extend(["--image-quality", str(img_q)])
            cmd.extend(["--image-scale", str(img_s)])
            img_dim = cfg_data.get("image_max_dim", args.image_max_dim)
            if img_dim:
                cmd.extend(["--image-max-dim", str(img_dim)])
        else:
            cmd.append("--no-compress-images")

        # 章節歸檔
        org_pages = cfg_data.get("organize_pages", args.organize_pages)
        if org_pages:
            cmd.append("--organize-pages")
            if cfg_data.get("copy", args.copy):
                cmd.append("--copy")
            org_target = cfg_data.get("organize_target", args.organize_target)
            cmd.extend(["--organize-target", org_target])
        else:
            cmd.append("--no-organize-pages")

        # 執行子進程
        book_start = time.time()
        print(f"[*] 調度指令: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            elapsed = time.time() - book_start
            print(f"\n[+] 書籍 '{title}' 處理完成！耗時: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分鐘)")
            results.append({
                "index": idx,
                "title": title,
                "directory": str(b_dir),
                "output_dir": str(out_sub_dir),
                "status": "SUCCESS",
                "elapsed": round(elapsed, 2),
                "error": None
            })
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - book_start
            print(f"\n[-] 錯誤：書籍 '{title}' 執行過程發生異常 (代碼: {e.returncode})", file=sys.stderr)
            results.append({
                "index": idx,
                "title": title,
                "directory": str(b_dir),
                "output_dir": str(out_sub_dir),
                "status": "FAILED",
                "elapsed": round(elapsed, 2),
                "error": f"CalledProcessError: {e}"
            })
            if args.stop_on_error or not args.continue_on_error:
                print("[-] 依 --stop-on-error 設定立即中止批次佇列！", file=sys.stderr)
                break
        except Exception as e:
            elapsed = time.time() - book_start
            print(f"\n[-] 錯誤：書籍 '{title}' 發生未預期例外: {e}", file=sys.stderr)
            results.append({
                "index": idx,
                "title": title,
                "directory": str(b_dir),
                "output_dir": str(out_sub_dir),
                "status": "FAILED",
                "elapsed": round(elapsed, 2),
                "error": str(e)
            })
            if args.stop_on_error or not args.continue_on_error:
                break

    # 5. 產出匯總報表
    total_time = time.time() - batch_start_time
    success_cnt = sum(1 for r in results if r["status"] == "SUCCESS")
    failed_cnt = sum(1 for r in results if r["status"] == "FAILED")
    skipped_cnt = sum(1 for r in results if r["status"] == "SKIPPED")

    report_data = {
        "generated_at": datetime.now().isoformat(),
        "total_books": len(book_tasks),
        "processed_books": len(results),
        "success_count": success_cnt,
        "failed_count": failed_cnt,
        "skipped_count": skipped_cnt,
        "total_elapsed_seconds": round(total_time, 2),
        "results": results
    }

    report_path = Path(args.report) if args.report else (output_root / "batch_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 同時產生 Markdown 摘要報告
    md_path = report_path.with_suffix(".md")
    md_lines = [
        "# ScanORC 批次數位化執行報告",
        f"- **執行時間**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **總書籍數**：{len(book_tasks)} 本",
        f"- **成功完成**：{success_cnt} 本",
        f"- **處理失敗**：{failed_cnt} 本",
        f"- **續傳略過**：{skipped_cnt} 本",
        f"- **總耗時**：{total_time:.1f} 秒 ({total_time / 60:.1f} 分鐘)",
        "",
        "## 書籍執行詳情",
        "| 編號 | 書名標題 | 狀態 | 耗時 (秒) | 成果目錄 |",
        "| :--- | :--- | :---: | :---: | :--- |",
    ]
    for r in results:
        status_icon = "✅ 成功" if r["status"] == "SUCCESS" else ("⏭ 略過" if r["status"] == "SKIPPED" else "❌ 失敗")
        md_lines.append(f"| {r['index']} | {r['title']} | {status_icon} | {r['elapsed']} | `{r['output_dir']}` |")
    md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n" + "#" * 70)
    print(" [🎉] ScanORC 批次書籍排程建置作業結束！")
    print(f"      總書籍數: {len(book_tasks)} | 成功: {success_cnt} | 失敗: {failed_cnt} | 略過: {skipped_cnt}")
    print(f"      總耗時: {total_time:.1f} 秒 ({total_time / 60:.1f} 分鐘)")
    print(f"      JSON 報告: {report_path}")
    print(f"      Markdown 報告: {md_path}")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
