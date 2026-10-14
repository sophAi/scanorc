#!/usr/bin/env python3
"""
ScanORC 目錄萃取與自動歸檔工具 (extract_toc.py)
功能：
  1. 支援傳入多個目錄圖檔或目錄路徑 (--input)，自動按檔名自然順序 (Natural Sort) 依序識別跨頁目錄
  2. 基於 GLM-OCR 0.9B 多模態模型，自動穿透底色色塊、智能過濾導引點線 (···· / ....)
  3. 結構化解析章節編號、標題、起始頁碼與結束頁碼，輸出標準 JSON 檔案
  4. 支援自動建立各章節對應目錄 (--organize-pages)，並將 Page-xxx.jpg 圖檔自動移入/複製至對應目錄中
"""

import os
import re
import sys
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from PIL import Image
from transformers import AutoProcessor
try:
    from transformers import GlmOcrForConditionalGeneration
except ImportError:
    try:
        from transformers import AutoModelForCausalLM as GlmOcrForConditionalGeneration
    except ImportError:
        GlmOcrForConditionalGeneration = None

from config_utils import attach_preference_and_template_support

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def natural_sort_key(p: Path) -> List[Any]:
    """
    檔名自然數字排序鍵 (例如 Cover-05 排在 Cover-06 前面，Page-2 排在 Page-10 前面)
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", p.name)]


def collect_toc_images(input_args: List[str]) -> List[Path]:
    """
    從命令列參數中收集所有目錄圖檔：支援單檔、多檔、萬用字元與目錄，並依自然順序排序
    """
    collected = []
    for arg in input_args:
        p = Path(arg).resolve()
        if p.is_file():
            if p.suffix.lower() in VALID_IMAGE_EXTS:
                collected.append(p)
            else:
                print(f"[-] 警告：略過非支援的圖片檔案 '{p.name}'", file=sys.stderr)
        elif p.is_dir():
            for root, _, files in os.walk(p):
                for f in files:
                    fp = Path(root) / f
                    if fp.suffix.lower() in VALID_IMAGE_EXTS:
                        collected.append(fp)
        else:
            # 支援 shell 未展開之萬用字元 (例如 "samples/Cover-0*.jpg")
            parent = p.parent
            pattern = p.name
            if parent.exists():
                matches = list(parent.glob(pattern))
                for m in matches:
                    if m.is_file() and m.suffix.lower() in VALID_IMAGE_EXTS:
                        collected.append(m)

    unique_images = list({img.resolve(): img for img in collected}.values())
    unique_images.sort(key=natural_sort_key)
    return unique_images


def find_default_content_images(search_dirs: Optional[List[Path]] = None) -> List[Path]:
    """
    預設自動搜尋檔名為 Content-xx.jpg 或 content-xx.jpg 的檔案群，並依照自然編號排序。
    搜尋範圍包含當前目錄、preprocessed、preprocessed_samples、samples、output/preprocessed 等。
    """
    if not search_dirs:
        search_dirs = [Path.cwd()]
        for d_name in ["preprocessed", "preprocessed_samples", "samples", "output/preprocessed", "output"]:
            d = Path.cwd() / d_name
            if d.exists() and d.is_dir():
                search_dirs.append(d)

    content_pattern = re.compile(r"^[Cc]ontent[-_]?\d+\.(jpg|jpeg|png|webp|tif|tiff)$", re.IGNORECASE)

    # 1. 優先在該目錄本層直接尋找 (避免混合多個子目錄)
    for d in search_dirs:
        if not d.exists() or not d.is_dir():
            continue
        direct_matches = [
            f for f in d.iterdir()
            if f.is_file() and content_pattern.match(f.name)
        ]
        if direct_matches:
            direct_matches.sort(key=natural_sort_key)
            return direct_matches

    # 2. 若當前層無匹配，搜尋子目錄 (排除隱藏與建置目錄)
    for d in search_dirs:
        if not d.exists() or not d.is_dir():
            continue
        for root, dirs, files in os.walk(d):
            dirs[:] = [dn for dn in dirs if not dn.startswith(".") and dn not in ("deb_dist", "node_modules", "dist", "build", "__pycache__")]
            sub_matches = [
                Path(root) / f for f in files
                if content_pattern.match(f)
            ]
            if sub_matches:
                sub_matches.sort(key=natural_sort_key)
                return sub_matches

    return []



def extract_toc_from_image(
    image_path: Path,
    processor: AutoProcessor,
    model: GlmOcrForConditionalGeneration,
    device: str,
    max_ocr_dim: int = 1280,
    max_tokens: int = 1024
) -> str:
    """
    使用 GLM-OCR 進行整頁文字識別
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_ocr_dim:
        img.thumbnail((max_ocr_dim, max_ocr_dim), Image.Resampling.LANCZOS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Text Recognition:"}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(images=[img], text=[prompt], return_tensors="pt").to(device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != torch.float16 and device.startswith("cuda"):
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)

    raw_text = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return raw_text


def parse_toc_text_raw(raw_text: str, source_filename: str = "") -> List[Dict[str, Any]]:
    """
    從 OCR 原始文本中抽取單行章節與頁碼項目
    """
    line_pattern = re.compile(r"^(?P<raw_title>.+?)\s*[·.．…\s\-]{2,}\s*(?P<page>\d+)\s*$")
    chapter_pattern = re.compile(r"^(第\s*\d+\s*[章節回卷部篇])\s*(.*)$")

    entries = []
    lines = raw_text.strip().splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line in ("目錄", "Contents", "目 錄", "CONTENTS"):
            continue

        match = line_pattern.match(line)
        if match:
            raw_title = match.group("raw_title").strip()
            page = int(match.group("page"))

            ch_match = chapter_pattern.match(raw_title)
            if ch_match:
                ch_num = re.sub(r"\s+", "", ch_match.group(1))
                ch_name = ch_match.group(2).strip()
            else:
                ch_num = re.sub(r"\s+", "", raw_title)
                ch_name = re.sub(r"\s+", "", raw_title)

            full_title = f"{ch_num} {ch_name}" if ch_num != ch_name else ch_name

            entries.append({
                "chapter": ch_num,
                "title": ch_name,
                "full_title": full_title,
                "start_page": page,
                "source_file": source_filename
            })

    return entries


def validate_toc(entries: List[Dict[str, Any]]) -> bool:
    """
    校驗頁碼單調遞增性
    """
    is_valid = True
    for i in range(len(entries) - 1):
        if entries[i]["start_page"] >= entries[i + 1]["start_page"]:
            print(
                f"[-] 警告：頁碼順序異常！條目 '{entries[i]['full_title']}' (頁 {entries[i]['start_page']}) "
                f">= 下一條目 '{entries[i+1]['full_title']}' (頁 {entries[i+1]['start_page']})",
                file=sys.stderr
            )
            is_valid = False
    return is_valid


def format_chapter_dirname(entry: Dict[str, Any], total_entries: int) -> str:
    """
    根據章節資訊格式化標準子目錄名稱 (例如 '01-一切從此開始', '07-放空_獲利捷徑')
    """
    ch_num = entry.get("chapter", "").strip()
    ch_title = entry.get("title", "").strip()
    index = entry.get("index", 1)

    match = re.search(r"第\s*(\d+)\s*[章節回卷部篇]", ch_num)
    if match:
        num = int(match.group(1))
        prefix = f"{num:02d}"
    else:
        prefix = f"{(index - 1):02d}"

    # 替換作業系統不允許或容易出錯的字元 (包含全半形冒號)
    clean_title = re.sub(r'[\/:*?"<>|：]', "_", ch_title)
    clean_title = re.sub(r"\s+", " ", clean_title).strip()

    if clean_title:
        return f"{prefix}-{clean_title}"
    return f"{prefix}-{ch_num}"


def find_matching_chapter(page_num: int, toc_entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    根據頁碼查詢屬於哪一個章節區間
    """
    for entry in toc_entries:
        s = entry["start_page"]
        e = entry["end_page"]
        if e is not None:
            if s <= page_num <= e:
                return entry
        else:
            if page_num >= s:
                return entry
    return None


def organize_pages_by_toc(
    pages_dir: Path,
    toc_entries: List[Dict[str, Any]],
    copy_mode: bool = False,
    exclude_files: Optional[List[Path]] = None
) -> Dict[str, int]:
    """
    自動建立各章節子目錄，並將 Page-xxx.jpg 等檔案歸檔至對應目錄中
    """
    exclude_set = {f.resolve() for f in (exclude_files or [])}
    action_name = "複製" if copy_mode else "移動"

    print("\n==================================================")
    print(f"[*] 執行掃描頁面自動歸檔 ({action_name}至章節目錄)")
    print(f"[*] 目標目錄: {pages_dir}")
    print("==================================================")

    # 預先建立所有章節子目錄
    dir_mapping = {}
    for entry in toc_entries:
        dirname = format_chapter_dirname(entry, len(toc_entries))
        target_dir = pages_dir / dirname
        target_dir.mkdir(parents=True, exist_ok=True)
        dir_mapping[entry["index"]] = target_dir

    # 掃描目標目錄根目錄下的所有圖檔
    root_files = [
        f for f in pages_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTS and f.resolve() not in exclude_set
    ]

    stats = {dirname: 0 for dirname in [d.name for d in dir_mapping.values()]}
    stats["[未配對]"] = 0
    moved_count = 0

    for f in root_files:
        # 從檔名中提取頁碼數字 (例如 Page-050.jpg -> 50, p020.png -> 20)
        nums = re.findall(r"\d+", f.stem)
        if not nums:
            continue

        page_num = int(nums[-1])
        target_chapter = find_matching_chapter(page_num, toc_entries)

        if target_chapter:
            dest_dir = dir_mapping[target_chapter["index"]]
            dest_file = dest_dir / f.name

            if copy_mode:
                shutil.copy2(f, dest_file)
            else:
                shutil.move(f, dest_file)

            stats[dest_dir.name] += 1
            moved_count += 1
        else:
            stats["[未配對]"] += 1

    print(f"[+] 歸檔完成！共處理 {moved_count} 個圖檔：")
    for d_name, count in stats.items():
        if count > 0:
            print(f"  - {d_name}: {count} 頁")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="ScanORC TOC Extractor: 掃描書本目錄頁文字與頁次結構化抽取工具 (支援多圖檔與自動建目錄歸檔)"
    )
    parser.add_argument(
        "--input", "-i",
        nargs="*",
        default=None,
        help="輸入目錄頁圖片路徑 (選填；預設自動搜尋 Content-xx.jpg 或 content-xx.jpg 檔案群並依編號排序，指定此參數則覆蓋預設搜尋)"
    )
    parser.add_argument(
        "--output", "-o",
        default="toc.json",
        help="輸出 JSON 檔案路徑 (預設: toc.json)"
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="AI 模型權重母目錄 (若指定則自動尋找 <models-dir>/glm-ocr；亦可透過 SCANORC_MODELS_DIR 設定)"
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="GLM-OCR 模型權重目錄 (若未指定則優先由 --models-dir 或 SCANORC_MODELS_DIR 推導，預設: models/glm-ocr)"
    )
    parser.add_argument(
        "--book-title",
        default="",
        help="書籍名稱 (選填，將記錄於 JSON metadata 中)"
    )
    parser.add_argument(
        "--organize-pages",
        metavar="DIR",
        nargs="?",
        const="",
        default=None,
        help="建立對應目錄的 directory，並將 Page-xxx.jpg 檔案移至對應的目錄裡。可指定圖檔目錄；若未指定路徑則預設整理輸入圖檔所在目錄"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="配合 --organize-pages 使用：複製檔案而非移動檔案 (預設為移動)"
    )
    args = attach_preference_and_template_support(parser, "extract_toc")

    # 判定輸入來源：若使用者未傳入 -i / --input，預設自動搜尋 Content-xx.jpg 或 content-xx.jpg
    if args.input:
        input_list = args.input if isinstance(args.input, list) else [args.input]
        is_auto_search = False
    else:
        auto_found = find_default_content_images()
        if not auto_found:
            print("[-] 錯誤：未在目錄中找到任何預設目錄圖檔 (Content-xx.jpg 或 content-xx.jpg)！", file=sys.stderr)
            print("    請確認目錄是否有 Content-01.jpg 等檔案，或透過 -i / --input 明確指定圖檔路徑。", file=sys.stderr)
            sys.exit(1)
        input_list = [str(p) for p in auto_found]
        is_auto_search = True

    json_candidates = [Path(a).resolve() for a in input_list if Path(a).is_file() and Path(a).suffix.lower() == ".json"]
    if json_candidates:
        json_file = json_candidates[0]
        print("==================================================")
        print(" ScanORC 書本目錄結構與自動歸檔工具 (載入現有 JSON)")
        print("==================================================")
        print(f"[*] 載入現有目錄索引 JSON 檔案: {json_file}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_entries = data.get("toc", [])
            if not all_entries:
                print(f"[-] 錯誤：'{json_file.name}' 中未找到任何目錄條目 (toc)！", file=sys.stderr)
                sys.exit(1)
            print(f"[+] 成功讀取 {len(all_entries)} 個章節目錄條目，無需加載模型推論！")
            sorted_images = []
        except Exception as e:
            print(f"[-] 讀取 JSON 檔案失敗: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        sorted_images = collect_toc_images(input_list)
        if not sorted_images:
            print(f"[-] 錯誤：未在指定的輸入路徑找到任何有效的圖片檔案！{input_list}", file=sys.stderr)
            sys.exit(1)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print("==================================================")
        print(" ScanORC 書本目錄結構化抽取工具 (方案 A: GLM-OCR)")
        print("==================================================")
        if is_auto_search:
            print(f"[*] 預設自動搜尋並鎖定 {len(sorted_images)} 個目錄頁 (Content-xx/content-xx)，已依編號自然排序：")
        else:
            print(f"[*] 自訂輸入覆蓋：發現 {len(sorted_images)} 個目錄圖檔，依檔名順序排列：")
        for idx, p in enumerate(sorted_images, 1):
            print(f"    ({idx}/{len(sorted_images)}) {p}")
        print(f"[*] 輸出位置: {args.output}")
        print(f"[*] 推論裝置: {device} ({dtype})")
        # 解析 GLM-OCR 模型路徑 (優先級: --model-dir > --models-dir/glm-ocr > SCANORC_MODELS_DIR > 專案內置 models)
        script_dir = Path(__file__).resolve().parent
        env_models_dir = os.environ.get("SCANORC_MODELS_DIR")
        if args.model_dir:
            actual_model_dir = Path(args.model_dir).resolve()
        elif args.models_dir:
            actual_model_dir = (Path(args.models_dir) / "glm-ocr").resolve()
        elif env_models_dir:
            actual_model_dir = (Path(env_models_dir) / "glm-ocr").resolve()
        elif (script_dir / "models" / "glm-ocr").exists():
            actual_model_dir = (script_dir / "models" / "glm-ocr").resolve()
        elif Path("/opt/models/glm-ocr").exists():
            actual_model_dir = Path("/opt/models/glm-ocr").resolve()
        else:
            cand = script_dir / "models" / "glm-ocr"
            actual_model_dir = cand.resolve() if cand.exists() else Path("models/glm-ocr").resolve()
        model_dir_str = str(actual_model_dir)

        print(f"[*] 載入模型: {model_dir_str} ...")

        processor = AutoProcessor.from_pretrained(model_dir_str, trust_remote_code=True)
        model = GlmOcrForConditionalGeneration.from_pretrained(
            model_dir_str,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True
        )
        model.eval()

        print(f"[+] 模型就緒！開始依序識別各目錄頁...")

        all_entries = []
        raw_texts = []

        for idx, img_path in enumerate(sorted_images, 1):
            print(f"\n[*] [{idx}/{len(sorted_images)}] 正在識別目錄頁: {img_path.name}...")
            raw_ocr_text = extract_toc_from_image(img_path, processor, model, device)
            raw_texts.append(f"--- File: {img_path.name} ---\n" + raw_ocr_text.strip())

            page_entries = parse_toc_text_raw(raw_ocr_text, source_filename=img_path.name)
            print(f"    [+] 從 {img_path.name} 解析出 {len(page_entries)} 個章節條目")
            all_entries.extend(page_entries)

        if not all_entries:
            print("[-] 警告：未能在所有目錄圖片中解析出任何有效目錄條目！", file=sys.stderr)
            sys.exit(1)

        # 全域計算結束頁碼與總頁數
        for i in range(len(all_entries)):
            all_entries[i]["index"] = i + 1
            if i < len(all_entries) - 1:
                end_page = all_entries[i + 1]["start_page"] - 1
                all_entries[i]["end_page"] = end_page
                all_entries[i]["page_count"] = max(1, end_page - all_entries[i]["start_page"] + 1)
            else:
                all_entries[i]["end_page"] = None
                all_entries[i]["page_count"] = None

        # 校驗頁碼連續性
        validate_toc(all_entries)

        # 封裝標準 JSON
        book_title = args.book_title
        if not book_title and sorted_images:
            p_dir = sorted_images[0].parent
            if p_dir.name.lower() in ("preprocessed", "preprocessed_samples", "output") and p_dir.parent.name:
                book_title = p_dir.parent.name
            else:
                book_title = p_dir.name or "Unknown"
        if not book_title:
            book_title = "Unknown"

        result_data = {
            "book_title": book_title,
            "source_images": [str(p) for p in sorted_images],
            "extracted_at": datetime.now().isoformat(),
            "model": model_dir_str,
            "total_chapters": len(all_entries),
            "toc": all_entries,
            "raw_text": "\n\n".join(raw_texts)
        }

        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        print("\n==================================================")
        print(f"[+] 跨頁目錄識別完成！全書共 {len(all_entries)} 個章節目錄條目")
        print(f"[+] JSON 成果已儲存至: {output_path}")

    # 印出預覽表格
    print("\n" + "=" * 68)
    print(f"{'索引':<4} {'章節編號':<8} {'章節標題':<28} {'起始頁':<8} {'結束頁':<8} {'總頁數'}")
    print("-" * 68)
    for item in all_entries:
        end_str = str(item['end_page']) if item['end_page'] is not None else "-"
        count_str = str(item['page_count']) if item['page_count'] is not None else "-"
        print(f"{item['index']:<4} {item['chapter']:<8} {item['title']:<28} {item['start_page']:<8} {end_str:<8} {count_str}")
    print("=" * 68 + "\n")

    # 執行自動建立目錄與移動 Page-xxx.jpg 歸檔
    if args.organize_pages is not None:
        if args.organize_pages:
            target_pages_dir = Path(args.organize_pages).resolve()
        elif sorted_images:
            target_pages_dir = sorted_images[0].parent
        else:
            target_pages_dir = Path.cwd()

        if not target_pages_dir.exists():
            print(f"[-] 錯誤：找不到待整理之目錄 '{target_pages_dir}'！", file=sys.stderr)
            sys.exit(1)

        organize_pages_by_toc(
            pages_dir=target_pages_dir,
            toc_entries=all_entries,
            copy_mode=args.copy,
            exclude_files=sorted_images
        )


if __name__ == "__main__":
    main()
