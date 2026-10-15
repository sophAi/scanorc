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

from config_utils import attach_preference_and_template_support

try:
    from orientation_detector import normalize_layout_direction
except ImportError:
    def normalize_layout_direction(val: Optional[str]) -> str:
        if not val:
            return "auto"
        s = str(val).strip().lower()
        if s in ("vertical", "vert", "vertical-rl", "v"):
            return "vertical"
        if s in ("horizontal", "horizon", "horizontal-tb", "h"):
            return "horizontal"
        return "auto"

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def natural_sort_key(p: Path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", p.name)]


def find_content_images(directory: Path) -> List[Path]:
    """
    在目錄中搜尋所有目錄頁圖檔 (Content-xx.jpg 或 content-xx.jpg 等)
    """
    content_pattern = re.compile(r"^[Cc]ontent[-_]?\d+\.(jpg|jpeg|png|webp|tif|tiff)$", re.IGNORECASE)
    contents = []
    for root, _, files in os.walk(directory):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in VALID_IMAGE_EXTS and content_pattern.match(p.name):
                contents.append(p)
    contents.sort(key=natural_sort_key)
    return contents


def find_cover01_image(directory: Path) -> Optional[Path]:
    """
    在目錄中搜尋預設封面圖檔 Cover-01.jpg
    """
    cover01_pattern = re.compile(r"^[Cc]over[-_]?0*1\.(jpg|jpeg|png|webp|tif|tiff)$", re.IGNORECASE)
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            p = Path(root) / f
            if p.suffix.lower() in VALID_IMAGE_EXTS and cover01_pattern.match(p.name):
                return p
    return None


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
        "--title", "-T",
        default="",
        help="書名 / 文檔標題 (若未指定則自動從輸入目錄名稱或 toc.json 讀取)"
    )
    parser.add_argument(
        "--toc",
        default=None,
        help="指定現有目錄結構 JSON 檔案路徑 (若未指定且輸入目錄中存在 toc.json，將自動優先沿用)"
    )
    parser.add_argument(
        "--force-toc",
        action="store_true",
        help="強制忽略既有 toc.json，一律重新以 AI 抽取目錄"
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
        default=1,
        help="版心四周均勻邊距大小 (單位: mm，預設: 1)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=50,
        help="轉正裁切若輸出為 JPG 時的壓縮品質 (1-100，預設: 50)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="轉正裁切輸出圖檔尺寸縮放比例 (例如 0.8 或 1.0，預設: 1.0)"
    )
    parser.add_argument(
        "--unsharp",
        default="1x0.55+1.5+0.002",
        help="轉正裁切影像銳化濾鏡參數 (ImageMagick unsharp 語法，例如 1x0.55+1.5+0.002，設為 none 或 off 關閉，預設: 1x0.55+1.5+0.002)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="GLM-OCR 生成文字時的重複懲罰係數 (預設: 1.1，防止引導線或符號陷入無限重複死循環)"
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=10,
        help="禁止連續重複的 n-gram 大小 (預設: 10，防止長引導點/破折號無限重複)"
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
        help="略過 OCR / ScanTailor 處理的檔案清單 (支援萬用字元如 Cover-*.jpg、Content-*、檔名、頁碼或 txt 清單，直接以原始圖檔導出並嵌入 HTML/EPUB)"
    )
    parser.add_argument(
        "--epub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否生成 MathML EPUB3 電子書 (預設: True)"
    )
    parser.add_argument(
        "--cover", "--cover-image",
        default=None,
        dest="cover",
        help="指定 EPUB 封面圖檔路徑 (若未指定將自動偵測 Cover-*.jpg 或 cover.* 圖檔)"
    )
    parser.add_argument(
        "--compress-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在編譯 EPUB 前，自動對 output/images 目錄中的 JPG 與 PNG 圖檔進行重新壓縮與尺寸縮放 (預設: True)"
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
        default=0.8,
        help="images 圖檔尺寸縮放比例 (例如 0.8 或 1.0，預設: 0.8)"
    )
    parser.add_argument(
        "--image-max-dim",
        type=int,
        default=None,
        help="images 圖檔最大長寬限制 (例如 1920，預設 None 不限制)"
    )
    parser.add_argument(
        "--image-unsharp",
        default=None,
        help="images 圖檔銳化參數 (ImageMagick unsharp 語法例如 1x0.55+1.5+0.002，預設: None 關閉)"
    )
    parser.add_argument(
        "--organize-pages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否執行章節目錄結構自動建立與圖檔歸檔 (預設: True)"
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
    parser.add_argument(
        "--models-dir",
        default=None,
        help="AI 模型權重母目錄 (若指定則傳遞至 extract_toc 與 pipeline；亦可透過 SCANORC_MODELS_DIR 設定)"
    )
    parser.add_argument(
        "--layout-direction",
        "--direction",
        dest="layout_direction",
        choices=["auto", "horizontal", "vertical", "horizon", "vertical-rl", "horizontal-tb"],
        default="auto",
        help="書籍排版方向 (auto: 優先讀取 toc.json，次之自動抽樣檢測, horizontal: 橫排, vertical: 直排, 預設: auto)"
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
    args = attach_preference_and_template_support(parser, "process_book")

    script_dir = Path(__file__).parent.resolve()
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.exists():
        print(f"[-] 錯誤：找不到輸入目錄 '{input_dir}'！", file=sys.stderr)
        sys.exit(1)

    book_title = args.title
    if not book_title:
        cov01 = find_cover01_image(input_dir)
        if cov01:
            cov_dir = cov01.resolve().parent
            if cov_dir.name.lower() in ("preprocessed", "preprocessed_samples", "output") and cov_dir.parent.name:
                book_title = cov_dir.parent.name
            else:
                book_title = cov_dir.name
    if not book_title:
        book_title = input_dir.name or "數位化掃描書籍"

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
    if args.quality:
        step1_cmd.extend(["--quality", str(args.quality)])
    if args.scale:
        step1_cmd.extend(["--scale", str(args.scale)])
    if args.unsharp:
        step1_cmd.extend(["--unsharp", str(args.unsharp)])
    if args.skip_ocr:
        step1_cmd.extend(["--skip-orc", *args.skip_ocr])

    print(f"[*] 執行指令: {' '.join(step1_cmd)}")
    subprocess.run(step1_cmd, check=True)

    # -------------------------------------------------------------------------
    # [步驟 2/4] 跨頁目錄識別與抽取 (extract_toc.py)
    # -------------------------------------------------------------------------
    print_banner(2, TOTAL_STEPS, "目錄頁次結構化抽取 (GLM-OCR -> toc.json)")
    has_toc = False
    existing_toc = None
    if not args.force_toc:
        if args.toc and Path(args.toc).exists():
            existing_toc = Path(args.toc).resolve()
        elif (input_dir / "toc.json").exists():
            existing_toc = (input_dir / "toc.json").resolve()
        elif toc_json_path.exists():
            existing_toc = toc_json_path.resolve()

    content_images: List[Path] = []
    if existing_toc:
        print(f"[+] 發現現有目錄結構索引: {existing_toc} (跳過 AI 抽取步驟)")
        if existing_toc != toc_json_path:
            shutil.copy2(existing_toc, toc_json_path)
            print(f"    已複製至成果目錄: {toc_json_path}")
        has_toc = True
    else:
        # 優先從轉正後目錄與輸入目錄搜尋 Content-xx.jpg / content-xx.jpg 目錄頁圖檔
        content_images = find_content_images(preprocessed_dir)
        if not content_images:
            content_images = find_content_images(input_dir)

        # 兼容性 fallback: 若無 Content-*.jpg，但有舊式多個 Cover-*.jpg 且非 Cover-01.jpg
        if not content_images:
            old_covers = find_cover_images(preprocessed_dir) or find_cover_images(input_dir)
            filtered = [p for p in old_covers if not re.match(r"^[Cc]over[-_]?0*1\.", p.name)]
            if filtered:
                content_images = filtered

        if content_images:
            print(f"[*] 發現 {len(content_images)} 個目錄頁圖檔：{[p.name for p in content_images]}")
            step2_cmd = [
                sys.executable, str(script_dir / "extract_toc.py"),
                "--input", *[str(p) for p in content_images],
                "--output", str(toc_json_path),
                "--book-title", book_title,
            ]
            if args.models_dir:
                step2_cmd.extend(["--models-dir", str(args.models_dir)])
            if args.layout_direction:
                step2_cmd.extend(["--layout-direction", str(args.layout_direction)])
            if args.repetition_penalty:
                step2_cmd.extend(["--repetition-penalty", str(args.repetition_penalty)])
            if args.no_repeat_ngram_size:
                step2_cmd.extend(["--no-repeat-ngram-size", str(args.no_repeat_ngram_size)])
            print(f"[*] 執行指令: {' '.join(step2_cmd)}")
            try:
                subprocess.run(step2_cmd, check=True)
                has_toc = toc_json_path.exists()
                if has_toc:
                    print(f"[+] 成功生成目錄結構索引 JSON: {toc_json_path}")
                else:
                    guess_path = toc_json_path.parent / "toc_guess.json"
                    if guess_path.exists():
                        print(f"[*] 提示：標準目錄未匹配，已自動產生候選目錄: {guess_path.name}")
                        print(f"    後續步驟將以無目錄模式繼續執行，待全書步驟完成後可人工校對另存為 {toc_json_path.name}。")
            except subprocess.CalledProcessError as e:
                print(f"[-] 警告：目錄結構抽取過程異常 ({e})，後續將以無目錄索引模式繼續執行", file=sys.stderr)
        else:
            print("[-] 未在輸入目錄中發現 Content-xx.jpg 目錄頁圖檔，跳過目錄抽取步驟")

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
    if not has_toc and toc_json_path.exists():
        has_toc = True
    if has_toc:
        step3_cmd.extend(["--toc", str(toc_json_path)])
    if args.epub:
        step3_cmd.append("--epub")
    if args.models_dir:
        step3_cmd.extend(["--models-dir", str(args.models_dir)])
    if args.repetition_penalty:
        step3_cmd.extend(["--repetition-penalty", str(args.repetition_penalty)])
    if args.no_repeat_ngram_size:
        step3_cmd.extend(["--no-repeat-ngram-size", str(args.no_repeat_ngram_size)])
    if args.skip_ocr:
        step3_cmd.extend(["--skip-ocr", *args.skip_ocr])

    # 自動偵測書籍排版方向 (優先繼承 toc.json，次之抽樣 Page-*.jpg)
    layout_dir = normalize_layout_direction(args.layout_direction)
    if layout_dir == "auto":
        # 尋找可用之 toc.json (步驟 2 產出之成果、現成 TOC、輸入目錄或轉正目錄)
        cand_tocs = []
        if toc_json_path.exists():
            cand_tocs.append(toc_json_path)
        if args.toc and Path(args.toc).exists():
            cand_tocs.append(Path(args.toc).resolve())
        if (input_dir / "toc.json").exists():
            cand_tocs.append((input_dir / "toc.json").resolve())
        if (preprocessed_dir / "toc.json").exists():
            cand_tocs.append((preprocessed_dir / "toc.json").resolve())

        for cand_t in cand_tocs:
            try:
                with open(cand_t, "r", encoding="utf-8") as f:
                    t_data = json.load(f)
                if t_data.get("layout_direction"):
                    norm = normalize_layout_direction(t_data["layout_direction"])
                    if norm != "auto":
                        layout_dir = norm
                        ori_desc = "直排 (豎排 / vertical-rl)" if layout_dir == "vertical" else "橫排 (horizontal)"
                        print(f"[*] 步驟 3 繼承目錄分析所得之書籍排版方向 ({cand_t.name}): '{layout_dir}' ({ori_desc})")
                        break
            except Exception:
                pass

        if layout_dir == "auto":
            try:
                from orientation_detector import detect_orientation
                detect_target = preprocessed_dir if preprocessed_dir.exists() and any(preprocessed_dir.iterdir()) else input_dir
                detected_dir = detect_orientation(detect_target, sample_count=5, short_label=True)
                norm = normalize_layout_direction(detected_dir)
                if norm != "auto":
                    layout_dir = norm
                ori_desc = "直排 (豎排 / vertical-rl)" if layout_dir == "vertical" else "橫排 (horizontal)"
                print(f"[*] 執行排版方向自動偵測 (抽樣 Page-*.jpg): 判定為 '{layout_dir}' ({ori_desc})")
            except Exception as e:
                print(f"[-] 自動排版方向偵測異常 ({e})，交由 pipeline 自動處理", file=sys.stderr)
                layout_dir = "auto"
    else:
        ori_desc = "直排 (豎排 / vertical-rl)" if layout_dir == "vertical" else "橫排 (horizontal)"
        print(f"[*] 手動指定書籍版面排版方向: '{layout_dir}' ({ori_desc})")

    if layout_dir:
        step3_cmd.extend(["--layout-direction", str(layout_dir)])

    # 封面圖檔設定與自動偵測 (優先尋找 Cover-01.jpg)
    actual_cover = args.cover
    if not actual_cover:
        actual_cover_path = find_cover01_image(preprocessed_dir) or find_cover01_image(input_dir)
        if actual_cover_path:
            actual_cover = str(actual_cover_path)
        else:
            for cand in [output_dir / "cover.jpg", input_dir / "cover.jpg", input_dir / "Cover.jpg"]:
                if cand.exists():
                    actual_cover = str(cand)
                    break
    if actual_cover and os.path.exists(actual_cover):
        step3_cmd.extend(["--cover", str(actual_cover)])
        print(f"[*] 設定 EPUB/HTML 封面圖檔: {actual_cover}")

    # images 圖檔重新壓縮與縮放設定
    if args.compress_images:
        step3_cmd.append("--compress-images")
        step3_cmd.extend(["--image-quality", str(args.image_quality)])
        step3_cmd.extend(["--image-scale", str(args.image_scale)])
        if args.image_max_dim:
            step3_cmd.extend(["--image-max-dim", str(args.image_max_dim)])
        if args.image_unsharp:
            step3_cmd.extend(["--image-unsharp", str(args.image_unsharp)])

    print(f"[*] 執行指令: {' '.join(step3_cmd)}")
    subprocess.run(step3_cmd, check=True)

    # -------------------------------------------------------------------------
    # [步驟 4/4] 根據 toc.json 生成章節目錄並將 Page-xxx.jpg 歸檔 (extract_toc.py --organize-pages)
    # -------------------------------------------------------------------------
    print_banner(4, TOTAL_STEPS, "章節目錄結構自動建立與圖檔歸檔")
    if args.organize_pages:
        target_dir = preprocessed_dir if args.organize_target == "preprocessed" else input_dir
        if toc_json_path.exists():
            print(f"[*] 準備對目錄進行章節歸檔: {target_dir} (依據現有目錄索引: {toc_json_path.name})")
            step4_cmd = [
                sys.executable, str(script_dir / "extract_toc.py"),
                "--input", str(toc_json_path),
                "--organize-pages", str(target_dir),
            ]
            if args.copy:
                step4_cmd.append("--copy")

            print(f"[*] 執行指令: {' '.join(step4_cmd)}")
            subprocess.run(step4_cmd, check=True)
            print(f"[+] 章節資料夾建立與圖檔歸檔完成！")
        else:
            print("[-] 由於缺少有效的 toc.json 目錄索引，略過章節歸檔步驟")
            guess_path = toc_json_path.parent / "toc_guess.json"
            if guess_path.exists():
                print(f"    提示：請於手動編修 {guess_path.name} 並另存為 {toc_json_path.name} 後，")
                print(f"    單獨執行以下指令進行章節歸檔：")
                print(f"    scanorc-toc -i {toc_json_path} --organize-pages {target_dir}")
    else:
        print("[*] 依設定略過章節目錄歸檔步驟 (--no-organize-pages)")

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
