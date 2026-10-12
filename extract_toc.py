#!/usr/bin/env python3
"""
ScanORC 目錄萃取工具 (extract_toc.py)
功能：
  1. 基於 GLM-OCR 0.9B 多模態模型，自動識別書本目錄頁 (TOC)
  2. 自動穿透底色色塊、智能過濾導引點線 (···· / ....)
  3. 結構化解析章節編號、標題、起始頁碼與結束頁碼
  4. 輸出標準 JSON 檔案，供後續 HTML / EPUB 流水線建立目錄索引與章節切分
"""

import os
import re
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, GlmOcrForConditionalGeneration


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


def parse_toc_text(raw_text: str) -> List[Dict[str, Any]]:
    """
    使用正規表達式解析 OCR 產出字串為結構化章節與頁碼
    """
    # 匹配模式：標題名稱 + 至少2個點/破折號/省略號 + 數字頁碼
    line_pattern = re.compile(r"^(?P<raw_title>.+?)\s*[·.．…\s\-]{2,}\s*(?P<page>\d+)\s*$")
    chapter_pattern = re.compile(r"^(第\s*\d+\s*[章節回卷部篇])\s*(.*)$")

    entries = []
    lines = raw_text.strip().splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 忽略目錄大標題
        if line in ("目錄", "Contents", "目 錄", "CONTENTS"):
            continue

        match = line_pattern.match(line)
        if match:
            raw_title = match.group("raw_title").strip()
            page = int(match.group("page"))

            # 拆分章節號與章節名稱 (例如 "第1章" 與 "一切從此開始")
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
                "start_page": page
            })

    # 自動推算各章結束頁碼與頁數
    for i in range(len(entries)):
        entries[i]["index"] = i + 1
        if i < len(entries) - 1:
            end_page = entries[i + 1]["start_page"] - 1
            entries[i]["end_page"] = end_page
            entries[i]["page_count"] = max(1, end_page - entries[i]["start_page"] + 1)
        else:
            entries[i]["end_page"] = None
            entries[i]["page_count"] = None

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


def main():
    parser = argparse.ArgumentParser(
        description="ScanORC TOC Extractor: 掃描書本目錄頁文字與頁次結構化抽取工具"
    )
    parser.add_argument(
        "--input", "-i",
        default="samples/Cover-06.jpg",
        help="輸入目錄頁圖片路徑 (預設: samples/Cover-06.jpg)"
    )
    parser.add_argument(
        "--output", "-o",
        default="toc.json",
        help="輸出 JSON 檔案路徑 (預設: toc.json)"
    )
    parser.add_argument(
        "--model-dir",
        default="models/glm-ocr",
        help="GLM-OCR 模型權重目錄 (預設: models/glm-ocr)"
    )
    parser.add_argument(
        "--book-title",
        default="",
        help="書籍名稱 (選填，將記錄於 JSON metadata 中)"
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"[-] 錯誤：找不到輸入圖片檔案 '{input_path}'！", file=sys.stderr)
        sys.exit(1)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("==================================================")
    print(" ScanORC 書本目錄結構化抽取工具 (方案 A: GLM-OCR)")
    print("==================================================")
    print(f"[*] 輸入圖片: {input_path}")
    print(f"[*] 輸出位置: {args.output}")
    print(f"[*] 推論裝置: {device} ({dtype})")
    print(f"[*] 載入模型: {args.model_dir} ...")

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = GlmOcrForConditionalGeneration.from_pretrained(
        args.model_dir,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    print(f"[+] 模型就緒！開始識別目錄頁...")
    raw_ocr_text = extract_toc_from_image(input_path, processor, model, device)

    print(f"[*] OCR 識別完成，正在解析章節與頁次結構...")
    toc_entries = parse_toc_text(raw_ocr_text)

    if not toc_entries:
        print("[-] 警告：未能在該圖片中解析出任何有效目錄條目！", file=sys.stderr)
        print("--- 原始識別文本 ---")
        print(raw_ocr_text)
        sys.exit(1)

    # 校驗頁碼連續性
    validate_toc(toc_entries)

    # 封裝標準 JSON
    book_title = args.book_title or input_path.parent.name or "Unknown"
    result_data = {
        "book_title": book_title,
        "source_image": str(input_path),
        "extracted_at": datetime.now().isoformat(),
        "total_chapters": len(toc_entries),
        "toc": toc_entries,
        "raw_text": raw_ocr_text.strip()
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"[+] 成功萃取 {len(toc_entries)} 個章節目錄條目！")
    print(f"[+] JSON 成果已儲存至: {output_path}")

    # 印出預覽表格
    print("\n" + "=" * 65)
    print(f"{'索引':<4} {'章節編號':<8} {'章節標題':<28} {'起始頁':<8} {'結束頁':<8} {'總頁數'}")
    print("-" * 65)
    for item in toc_entries:
        end_str = str(item['end_page']) if item['end_page'] is not None else "-"
        count_str = str(item['page_count']) if item['page_count'] is not None else "-"
        print(f"{item['index']:<4} {item['chapter']:<8} {item['title']:<28} {item['start_page']:<8} {end_str:<8} {count_str}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
