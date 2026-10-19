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
from typing import List, Dict, Any, Optional, Tuple

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
try:
    from orientation_detector import detect_orientation, detect_book_orientation
except ImportError:
    detect_orientation = None
    detect_book_orientation = None

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}

CN_NUM_MAP = {
    '零': 0, '一': 1, '二': 2, '兩': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
}


def parse_chinese_number(cn_str: str) -> Optional[int]:
    """將簡單中文數字 (一至九十九) 轉為整數"""
    cn_str = cn_str.strip()
    if not cn_str:
        return None
    if cn_str.isdigit():
        return int(cn_str)

    if cn_str == "十":
        return 10
    if cn_str.startswith("十"):
        return 10 + CN_NUM_MAP.get(cn_str[1:], 0)
    if "十" in cn_str:
        parts = cn_str.split("十")
        ten = CN_NUM_MAP.get(parts[0], 1) * 10
        one = CN_NUM_MAP.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return ten + one
    if len(cn_str) == 1 and cn_str in CN_NUM_MAP:
        return CN_NUM_MAP[cn_str]
    return None


def natural_sort_key(p: Path) -> List[Any]:
    """
    檔名自然數字排序鍵 (例如 Cover-05 排在 Cover-06 前面，Page-2 排在 Page-10 前面)
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", p.name)]


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


def collect_toc_images(input_args: List[str]) -> List[Path]:
    """
    從命令列參數中收集所有目錄圖檔：支援單檔、多檔、萬用字元與目錄，並依自然順序排序。
    若傳入參數為目錄，優先自動搜尋該目錄下之 Content-*.jpg 目錄圖檔。
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
            content_in_dir = find_default_content_images([p])
            if content_in_dir:
                collected.extend(content_in_dir)
            else:
                for root, dirs, files in os.walk(p):
                    dirs[:] = [dn for dn in dirs if not dn.startswith(".") and dn not in ("deb_dist", "node_modules", "dist", "build", "__pycache__")]
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



def extract_toc_from_image(
    image_path: Path,
    processor: AutoProcessor,
    model: GlmOcrForConditionalGeneration,
    device: str,
    max_ocr_dim: int = 1280,
    max_tokens: int = 1024,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 10
) -> str:
    """
    使用 GLM-OCR 進行整頁文字識別 (支援重複懲罰與 n-gram 限制防止長點線/破折號無限死循環)
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

    gen_kwargs = {
        "max_new_tokens": max_tokens,
    }
    if repetition_penalty is not None and repetition_penalty > 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    raw_text = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return raw_text


def parse_toc_text_raw(raw_text: str, source_filename: str = "") -> List[Dict[str, Any]]:
    """
    從 OCR 原始文本中抽取章節與頁碼項目。
    支援：
    1. 傳統引導點線 (......) 或破折號 (—— / — / ― / ─) 或空格分隔之頁碼
    2. 多行/跨行分離頁碼模式 (第一行標題，次行純頁碼數字，自動配對合併)
    3. 多行長標題自動合併 (章節名無頁碼時，自動串接次行次標題與頁碼)
    4. 多種常見中文及英文章節、序言、數字前綴 (第X章, Part X, Chapter X, 1., 01., 推薦序, 簡介等)
    5. 自動過濾句尾標點之裝飾引言/引用句
    """
    chapter_prefix_pat = re.compile(
        r"^(?P<prefix>"
        r"(?:第\s*([0-9]+|[一二三四五六七八九十百千兩]+)\s*[章節回卷部篇])|"
        r"(?:(?:Chapter|Part|Section|Unit)\s*([0-9]+|[A-Za-z]+|[IVXLCDM]+))|"
        r"(?:(?:[0-9]{1,3}|[一二三四五六七八九十]+)[\.．、\s]+)|"
        r"(?:[\(（](?:[0-9]{1,3}|[一二三四五六七八九十]+)[\)）]\s*)|"
        r"(?:推薦序|自序|總序|譯序|作者序|編者序|序言|序|前言|導讀|簡介|引言|跋|後記|附錄|結語|尾聲|附錄[A-Za-z0-9一二三四五六七八九十]*)"
        r")\s*(?P<remainder>.*)$",
        re.IGNORECASE
    )
    # 匹配行尾數字作為頁碼：允許 1 個以上引導符號 (含點、破折號、波浪線等) 或 1 個以上空格
    line_pattern = re.compile(r"^(?P<raw_title>.+?)\s*(?:[·.．…・\-_|—―─~～‧•]{1,}|\s+)\s*(?P<page>\d{1,4})\s*$")
    # 匹配獨立純頁碼行 (例如 '123' 或 'P123')
    pure_page_pat = re.compile(r"^[\(\[\{Pｐ頁]?\s*(\d{1,4})\s*[\)\]\}頁]?$")

    entries = []
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    pending = None

    def finalize_pending():
        nonlocal pending
        if not pending:
            return
        ch = pending["chapter"].strip()
        parts = [p.strip() for p in pending["title_parts"] if p.strip()]
        if not parts:
            raw_ch_title = ch
        else:
            combined = parts[0]
            for next_p in parts[1:]:
                if combined and combined[-1] in ("；", "：", "，", "、", ";", ":", ","):
                    combined += next_p
                else:
                    combined += " " + next_p
            raw_ch_title = combined.strip()

        # 清除首尾殘留的引導點線、破折號與無效符號
        raw_ch_title = re.sub(r"^[\s·.．…・\-_|—―─~～‧•]+", "", raw_ch_title).strip()
        raw_ch_title = re.sub(r"[\s·.．…・\-_|—―─~～‧•]+$", "", raw_ch_title).strip()

        # 若既無章節前綴，又無有效頁碼，捨棄 (避免孤立廣告標語或引言變成條目)
        if not ch and pending["page"] is None:
            pending = None
            return

        full_title = f"{ch} {raw_ch_title}".strip() if ch and raw_ch_title != ch else (raw_ch_title or ch)

        entries.append({
            "chapter": ch,
            "title": raw_ch_title,
            "full_title": full_title,
            "start_page": pending["page"],
            "source_file": source_filename
        })
        pending = None

    for line in lines:
        if line.startswith("--- File:"):
            continue
        if line in ("目錄", "Contents", "目 錄", "CONTENTS", "目　錄"):
            continue

        m_pure_page = pure_page_pat.match(line)
        m_end_page = line_pattern.match(line)
        m_prefix = chapter_prefix_pat.match(line)

        # 1. 純數字行 (跨行分離之頁碼行)
        if m_pure_page and pending and pending["page"] is None:
            pending["page"] = int(m_pure_page.group(1))
            finalize_pending()
            continue

        # 2. 行內直接帶有頁碼 (如 "標題 ...... 105" 或 "標題——105")
        if m_end_page:
            raw_title = m_end_page.group("raw_title").strip()
            raw_title = re.sub(r"[\s·.．…・\-_|—―─~～‧•]+$", "", raw_title).strip()
            page = int(m_end_page.group("page"))

            # 檢查此行本身是否含有章節前綴
            ch_m = chapter_prefix_pat.match(raw_title)
            if ch_m:
                # 獨立的新章節行 (具前綴、標題、頁碼)
                finalize_pending()
                ch_str = ch_m.group("prefix").strip()
                rem_title = ch_m.group("remainder").strip()
                pending = {
                    "chapter": ch_str,
                    "title_parts": [rem_title] if rem_title else [],
                    "page": page
                }
                finalize_pending()
            elif pending and not pending["title_parts"]:
                # pending 只有章節前綴，此行補充次標題與頁碼
                pending["title_parts"].append(raw_title)
                pending["page"] = page
                finalize_pending()
            else:
                # 無章節前綴但有頁碼之獨立條目 (如 "成功故事回饋 505" 或一般節)
                finalize_pending()
                pending = {
                    "chapter": "",
                    "title_parts": [raw_title],
                    "page": page
                }
                finalize_pending()

        # 3. 開頭為章節前綴，但行尾無頁碼 (例如多行標題的第一行或大章節名稱)
        elif m_prefix:
            finalize_pending()
            ch_str = m_prefix.group("prefix").strip()
            rem_title = m_prefix.group("remainder").strip()
            pending = {
                "chapter": ch_str,
                "title_parts": [rem_title] if rem_title else [],
                "page": None
            }

        # 4. 裝飾引言/引用句 (以句號結尾且有一定長度，略過)
        elif line.endswith(("。", "…", "！", "!", "？", "?")) and len(line) > 10:
            continue

        # 5. 一般未帶頁碼標題行 (次行可能會有純頁碼行)
        elif pending and not pending["title_parts"]:
            pending["title_parts"].append(line)
        else:
            finalize_pending()
            pending = {
                "chapter": "",
                "title_parts": [line],
                "page": None
            }

    finalize_pending()
    return entries



def extract_guess_toc(
    raw_texts: List[str],
    source_images: List[Path],
    resolved_layout_direction: str,
    model_dir_str: str,
    book_title: str,
    target_output_path: Path,
) -> Tuple[Path, List[Dict[str, Any]]]:
    """
    當標準目錄提取正則未能匹配任何條目時，觸發候選目錄生成模式 (TOC Guess Mode)。
    將 OCR 識別所得之每一行都當作一個條目，搜尋該行數字作為頁數 (start_page)；
    若無數字則寫 "none"。
    儲存至 target_output_path 同層目錄之 toc_guess.json，不覆蓋也不生成 toc.json。
    """
    guess_entries = []

    for img_path, raw_block in zip(source_images, raw_texts):
        lines = [line.strip() for line in raw_block.splitlines() if line.strip()]
        for line in lines:
            if line.startswith("--- File:"):
                continue
            if line in ("目錄", "Contents", "目 錄", "CONTENTS", "目　錄"):
                continue

            # 辨識章節前綴 (例如 '第一章', '第1章', 'Chapter 1' 等)
            ch_match = re.match(r"^(第\s*([0-9]+|[一二三四五六七八九十百千兩]+)\s*[章節回卷部篇])\s*(.*)$", line)
            if ch_match:
                ch_num = re.sub(r"\s+", "", ch_match.group(1))
                remainder = ch_match.group(3).strip()
            else:
                ch_num = ""
                remainder = line

            # 搜尋該行數字的部分當作頁數
            # 優先搜尋行尾數字 (例如 '投資心得 ...... 15' 或 '投資心得 15')
            tail_match = re.search(r"(\d+)\s*$", remainder or line)
            all_digits = re.findall(r"\b\d+\b", remainder)

            if tail_match:
                start_page = int(tail_match.group(1))
                clean_title = re.sub(r"[\s·.．…・\-_|—―─~～‧•]+" + tail_match.group(1) + r"\s*$", "", remainder or line).strip()
            elif all_digits:
                start_page = int(all_digits[-1])
                clean_title = re.sub(r"[\s·.．…・\-_|—―─~～‧•]+" + str(start_page) + r"\b", "", remainder or line).strip()
            else:
                start_page = "none"
                clean_title = remainder or line

            # 清理引導點線與無效符號
            clean_title = re.sub(r"[·.．…・\-_|—―─~～‧•]{2,}", "", clean_title).strip()
            if not clean_title and ch_num:
                clean_title = ch_num

            full_title = f"{ch_num} {clean_title}".strip() if ch_num and clean_title != ch_num else (clean_title or ch_num or line)

            guess_entries.append({
                "index": len(guess_entries) + 1,
                "chapter": ch_num,
                "title": clean_title,
                "full_title": full_title,
                "start_page": start_page,
                "end_page": "none",
                "page_count": "none",
                "source_file": img_path.name
            })

    guess_output_path = target_output_path.parent / "toc_guess.json"
    guess_output_path.parent.mkdir(parents=True, exist_ok=True)

    guess_data = {
        "book_title": book_title,
        "layout_direction": resolved_layout_direction,
        "is_guess": True,
        "note": "此檔案由 extract_toc.py 於標準目錄格式未匹配時自動生成之候選目錄 (toc_guess.json)。請人工校對並補全頁碼後，另存為 toc.json 供後續生成 EPUB/HTML 使用。",
        "source_images": [str(p) for p in source_images],
        "extracted_at": datetime.now().isoformat(),
        "model": model_dir_str,
        "total_chapters": len(guess_entries),
        "toc": guess_entries,
        "raw_text": "\n\n".join(raw_texts)
    }

    with open(guess_output_path, "w", encoding="utf-8") as f:
        json.dump(guess_data, f, ensure_ascii=False, indent=2)

    return guess_output_path, guess_entries


def validate_toc(entries: List[Dict[str, Any]]) -> bool:
    """
    校驗頁碼單調遞增性
    """
    is_valid = True
    for i in range(len(entries) - 1):
        p1 = entries[i].get("start_page")
        p2 = entries[i + 1].get("start_page")
        if p1 is None or p2 is None or str(p1).lower() == "none" or str(p2).lower() == "none":
            continue
        try:
            if int(p1) >= int(p2):
                print(
                    f"[-] 警告：頁碼順序異常！條目 '{entries[i]['full_title']}' (頁 {p1}) "
                    f">= 下一條目 '{entries[i+1]['full_title']}' (頁 {p2})",
                    file=sys.stderr
                )
                is_valid = False
        except (ValueError, TypeError):
            pass
    return is_valid


def format_chapter_dirname(entry: Dict[str, Any], total_entries: int) -> str:
    """
    根據章節資訊格式化標準子目錄名稱 (例如 '01-前言', '02-一切從此開始', '07-放空_獲利捷徑')
    使用 index 確保章節資料夾前綴序號絕對唯一且單調遞增，完全避免章節重疊覆蓋或排序錯亂
    """
    ch_num = entry.get("chapter", "").strip()
    ch_title = entry.get("title", "").strip()
    index = entry.get("index", 1)
    digits = max(2, len(str(total_entries)))
    prefix = f"{index:0{digits}d}"

    clean_title = re.sub(r'[\/:*?"<>|：，,;；]', "_", ch_title)
    clean_title = re.sub(r"\s+", " ", clean_title).strip()

    if clean_title:
        return f"{prefix}-{clean_title}"
    clean_num = re.sub(r'[\/:*?"<>|：，,;；]', "_", ch_num).strip()
    if clean_num:
        return f"{prefix}-{clean_num}"
    return f"{prefix}-第{index}章"


def find_matching_chapter(page_num: int, toc_entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    根據頁碼查詢屬於哪一個章節區間
    """
    for entry in toc_entries:
        s_raw = entry.get("start_page")
        e_raw = entry.get("end_page")
        if s_raw is None or str(s_raw).lower() == "none":
            continue
        try:
            s = int(s_raw)
            if e_raw is not None and str(e_raw).lower() != "none":
                e = int(e_raw)
                if s <= page_num <= e:
                    return entry
            else:
                if page_num >= s:
                    return entry
        except (ValueError, TypeError):
            continue
    return None


def organize_pages_by_toc(
    pages_dir: Path,
    toc_entries: List[Dict[str, Any]],
    copy_mode: bool = False,
    exclude_files: Optional[List[Path]] = None
) -> Dict[str, int]:
    """
    自動建立各章節子目錄，並將 Page-xxx.jpg 等檔案歸檔至對應目錄中。
    支援遞迴搜尋 pages_dir 及其所有子目錄中的圖檔，無論初始為平鋪或鏡像子目錄皆能正確整理歸檔。
    """
    exclude_set = {f.resolve() for f in (exclude_files or [])}
    action_name = "複製" if copy_mode else "移動"

    print("\n==================================================")
    print(f"[*] 執行掃描頁面自動歸檔 ({action_name}至章節目錄)")
    print(f"[*] 目標目錄: {pages_dir}")
    print("==================================================")

    # 建立所有章節子目錄對應表 (延遲建立目錄，僅在有圖檔歸入時建立)
    dir_mapping = {}
    for entry in toc_entries:
        dirname = format_chapter_dirname(entry, len(toc_entries))
        dir_mapping[entry["index"]] = pages_dir / dirname

    # 遞迴搜尋目標目錄下所有圖檔 (支援根目錄平鋪與子目錄分組結構)
    candidate_files = []
    for root, _, files in os.walk(pages_dir):
        for fname in sorted(files):
            f = Path(root) / fname
            if f.suffix.lower() not in VALID_IMAGE_EXTS:
                continue
            if f.resolve() in exclude_set:
                continue

            # 排除非書籍主體資料夾 (如 raw_pages 或 images)
            if any(part in ("raw_pages", "images", "deb_dist", ".git") for part in f.parts):
                continue

            f_lower = fname.lower()
            # 封面與目錄圖檔應保留於書籍根目錄中
            if re.search(r"^(cover|content|front[-_]?cover|封面)", f_lower):
                if f.parent.resolve() != pages_dir.resolve():
                    root_dest = pages_dir / f.name
                    if not root_dest.exists():
                        if copy_mode:
                            shutil.copy2(f, root_dest)
                        else:
                            shutil.move(str(f), str(root_dest))
                continue

            if f.stem.startswith("."):
                continue

            candidate_files.append(f)

    # 依檔名自然數字排序
    candidate_files.sort(
        key=lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]
    )

    stats = {d.name: 0 for d in dir_mapping.values()}
    stats["[未配對]"] = 0
    moved_count = 0
    already_placed_count = 0

    for f in candidate_files:
        # 從檔名中提取頁碼數字 (例如 Page-050.jpg -> 50, p020.png -> 20)
        nums = re.findall(r"\d+", f.stem)
        if not nums:
            continue

        page_num = int(nums[-1])
        target_chapter = find_matching_chapter(page_num, toc_entries)

        if target_chapter:
            dest_dir = dir_mapping[target_chapter["index"]]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / f.name

            d_name = dest_dir.name
            if f.resolve() == dest_file.resolve():
                stats[d_name] = stats.get(d_name, 0) + 1
                already_placed_count += 1
            else:
                if copy_mode:
                    shutil.copy2(f, dest_file)
                else:
                    shutil.move(str(f), str(dest_file))
                stats[d_name] = stats.get(d_name, 0) + 1
                moved_count += 1
        else:
            stats["[未配對]"] = stats.get("[未配對]", 0) + 1

    # 清理所有空資料夾 (包括全數移出圖檔之舊子目錄，或未歸入任何圖檔之章節目錄)
    for root, dirs, files in os.walk(pages_dir, topdown=False):
        p_root = Path(root)
        if p_root.resolve() == pages_dir.resolve():
            continue
        try:
            if not any(p_root.iterdir()):
                p_root.rmdir()
        except Exception:
            pass

    total_processed = moved_count + already_placed_count
    print(f"[+] 歸檔完成！共處理 {total_processed} 個圖檔 (新歸檔: {moved_count}, 已就位: {already_placed_count}, 未配對: {stats['[未配對]']})：")
    for d_name, count in sorted(stats.items()):
        if count > 0 and d_name != "[未配對]":
            print(f"  - {d_name}: {count} 頁")
    if stats["[未配對]"] > 0:
        print(f"  [!] 提示: 有 {stats['[未配對]']} 個圖檔未配對至任何章節區間，已保留原位")

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
        "--layout-direction",
        choices=["auto", "horizontal", "vertical", "horizon"],
        default="auto",
        help="指定目錄排版方向 (auto: 依 Content-*.jpg 目錄圖檔自動判斷, horizontal: 橫排, vertical: 直排；預設 auto)"
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
        "--auto-accept-guess",
        dest="auto_accept_guess",
        action="store_true",
        default=True,
        help="當候選目錄 (toc_guess) 品質符合門檻 (頁碼覆蓋率>=60%且順序正確) 時自動採納並寫入 toc.json (預設開啟)"
    )
    parser.add_argument(
        "--no-auto-accept-guess",
        dest="auto_accept_guess",
        action="store_false",
        help="停用候選目錄自動採納，強制僅輸出 toc_guess.json"
    )
    parser.add_argument(
        "--accept-guess",
        action="store_true",
        help="強制採納候選目錄 (toc_guess) 為正式 toc.json"
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
            resolved_layout_direction = data.get("layout_direction", "horizontal")
            if args.layout_direction != "auto":
                resolved_layout_direction = "horizontal" if args.layout_direction in ("horizon", "horizontal") else "vertical"
            ori_text = "直排 (豎排 / vertical-rl)" if resolved_layout_direction == "vertical" else "橫排 (horizontal)"
            print(f"[*] 目錄排版方向: {ori_text}")
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

        # ---------------------------------------------------------------------
        # 排版方向判斷：若為 auto 且有目錄圖檔，以 Content-*.jpg 判定
        # ---------------------------------------------------------------------
        if args.layout_direction == "auto":
            if detect_orientation is not None and sorted_images:
                detected_direction = detect_orientation(sorted_images, sample_count=len(sorted_images), short_label=True)
                resolved_layout_direction = "horizontal" if detected_direction in ("horizon", "horizontal") else "vertical"
                ori_text = "直排 (豎排 / vertical-rl)" if resolved_layout_direction == "vertical" else "橫排 (horizontal)"
                print(f"[*] 依據目錄圖檔 (Content-*.jpg) 自動判定排版方向: {ori_text}")
            else:
                resolved_layout_direction = "horizontal"
                print("[*] 目錄預設為橫排 (horizontal)")
        else:
            resolved_layout_direction = "horizontal" if args.layout_direction in ("horizon", "horizontal") else "vertical"
            ori_text = "直排 (豎排 / vertical-rl)" if resolved_layout_direction == "vertical" else "橫排 (horizontal)"
            print(f"[*] 指定目錄排版方向: {ori_text}")

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
            raw_ocr_text = extract_toc_from_image(
                img_path,
                processor,
                model,
                device,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size
            )
            raw_texts.append(f"--- File: {img_path.name} ---\n" + raw_ocr_text.strip())

            page_entries = parse_toc_text_raw(raw_ocr_text, source_filename=img_path.name)

            # 若該頁有章節條目未能識別出起始頁碼 (例如直排目錄頁碼位於欄位最底端，與標題距離過遠未被整頁 OCR 捕獲)
            missing_page_entries = [e for e in page_entries if e.get("start_page") is None]
            if missing_page_entries:
                try:
                    img_obj = Image.open(img_path).convert("RGB")
                    w, h = img_obj.size
                    bottom_crop = img_obj.crop((0, int(h * 0.70), w, h))
                    crop_messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": "Text Recognition:"}
                            ]
                        }
                    ]
                    crop_prompt = processor.apply_chat_template(crop_messages, add_generation_prompt=True)
                    crop_inputs = processor(images=[bottom_crop], text=[crop_prompt], return_tensors="pt").to(device)
                    if "pixel_values" in crop_inputs and crop_inputs["pixel_values"].dtype != torch.float16 and device.startswith("cuda"):
                        crop_inputs["pixel_values"] = crop_inputs["pixel_values"].to(torch.float16)
                    with torch.no_grad():
                        crop_outputs = model.generate(**crop_inputs, max_new_tokens=100)
                    crop_text = processor.decode(crop_outputs[0][crop_inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    crop_numbers = [int(n) for n in re.findall(r"\b\d{1,4}\b", crop_text)]
                    existing_pages = {e["start_page"] for e in page_entries if isinstance(e.get("start_page"), int)}
                    cand_pages = [p for p in crop_numbers if p not in existing_pages and p > 0]
                    if cand_pages:
                        for m_entry, cand_p in zip(missing_page_entries, sorted(cand_pages)):
                            m_entry["start_page"] = cand_p
                            print(f"    [+] 從頁尾補全章節 '{m_entry['full_title']}' 起始頁碼: {cand_p}")
                except Exception as ex:
                    print(f"    [-] 頁尾頁碼補充偵測略過: {ex}", file=sys.stderr)

            print(f"    [+] 從 {img_path.name} 解析出 {len(page_entries)} 個章節條目")
            all_entries.extend(page_entries)

        if not all_entries:
            print("\n[-] 提示：未能從目錄頁中匹配到標準章節與頁碼引導格式。")
            print("[*] 自動啟動候選目錄分析模式 (TOC Guess Mode)...")
            output_path = Path(args.output).resolve()
            book_title = args.book_title
            if not book_title and sorted_images:
                p_dir = sorted_images[0].parent
                if p_dir.name.lower() in ("preprocessed", "preprocessed_samples", "output") and p_dir.parent.name:
                    book_title = p_dir.parent.name
                else:
                    book_title = p_dir.name or "Unknown"
            if not book_title:
                book_title = "Unknown"

            guess_file, guess_entries = extract_guess_toc(
                raw_texts=raw_texts,
                source_images=sorted_images,
                resolved_layout_direction=resolved_layout_direction,
                model_dir_str=model_dir_str,
                book_title=book_title,
                target_output_path=output_path,
            )
            print(f"[+] 候選條目抽取完成！共整理出 {len(guess_entries)} 個候選條目")
            print(f"[+] 候選目錄檔案已儲存至: {guess_file}")

            # 評估候選目錄品質：若多數條目具有有效頁碼且具備單調性，自動採納
            total_guess = len(guess_entries)
            valid_page_entries = [e for e in guess_entries if isinstance(e.get("start_page"), int) and e.get("start_page") > 0]
            valid_ratio = len(valid_page_entries) / total_guess if total_guess > 0 else 0.0

            should_accept = False
            if getattr(args, "accept_guess", False):
                should_accept = True
                print("[*] 使用者指定 --accept-guess，強制採納候選目錄！")
            elif getattr(args, "auto_accept_guess", True) and len(valid_page_entries) >= 2 and valid_ratio >= 0.6:
                inc_count = sum(1 for k in range(len(valid_page_entries) - 1) if valid_page_entries[k]["start_page"] <= valid_page_entries[k + 1]["start_page"])
                monotonic_ratio = inc_count / (len(valid_page_entries) - 1) if len(valid_page_entries) > 1 else 1.0
                if monotonic_ratio >= 0.7:
                    should_accept = True
                    print(f"[+] 候選目錄品質良好 (頁碼覆蓋率: {valid_ratio:.0%}，順序單調性: {monotonic_ratio:.0%})，自動採納為正式目錄！")

            if should_accept:
                all_entries = guess_entries
            else:
                print(f"[*] 注意：候選目錄品質未達自動採納門檻，本步驟未產生 {output_path.name}，後續流程將以無目錄模式繼續執行。")
                print(f"    待全書轉檔完成後，您可以手動編輯 {guess_file.name} 並另存為 {output_path.name}，再重新編譯 HTML/EPUB。")

                # 印出候選預覽表格
                print("\n" + "=" * 68)
                print(f"{'索引':<4} {'章節編號':<8} {'章節標題':<28} {'起始頁':<8} {'結束頁':<8} {'總頁數'}")
                print("-" * 68)
                for item in guess_entries:
                    start_str = str(item['start_page'])
                    end_str = str(item['end_page'])
                    count_str = str(item['page_count'])
                    ch_str = str(item.get('chapter', ''))
                    title_str = str(item.get('title', ''))
                    print(f"{item['index']:<4} {ch_str:<8} {title_str:<28} {start_str:<8} {end_str:<8} {count_str}")
                print("=" * 68 + "\n")
                return

        # 直排目錄條目閱讀順序校驗：
        # 直式中文目錄通常由右向左 (RTL) 排列；若 OCR 引擎以左到右 (LTR) 掃描輸出，會導致章節起始頁碼呈現單調遞減。
        # 此時自動反轉列表為正確的由右至左 (RTL) 順序。
        valid_int_pages = [e["start_page"] for e in all_entries if isinstance(e.get("start_page"), int) and e["start_page"] > 0]
        if resolved_layout_direction == "vertical" and len(valid_int_pages) >= 2:
            descending_count = sum(
                1 for i in range(len(valid_int_pages) - 1)
                if valid_int_pages[i] > valid_int_pages[i + 1]
            )
            if descending_count > (len(valid_int_pages) - 1) * 0.7:
                print(f"[*] 偵測到直排目錄條目以從左至右 (LTR) 逆序排列 (倒序比例: {descending_count}/{len(valid_int_pages)-1})，自動反轉修正為由右至左 (RTL) 正確閱讀順序！")
                all_entries.reverse()

        # 全域計算結束頁碼與總頁數
        for i in range(len(all_entries)):
            all_entries[i]["index"] = i + 1
            curr_p = all_entries[i].get("start_page")

            # 尋找下一個具有有效整數頁碼的條目以決定結束頁
            next_p = None
            for j in range(i + 1, len(all_entries)):
                pj = all_entries[j].get("start_page")
                if isinstance(pj, int) and pj > 0:
                    next_p = pj
                    break

            if isinstance(curr_p, int) and curr_p > 0 and next_p is not None:
                end_page = next_p - 1
                all_entries[i]["end_page"] = max(curr_p, end_page)
                all_entries[i]["page_count"] = max(1, end_page - curr_p + 1)
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
            "layout_direction": resolved_layout_direction,
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
    for idx, item in enumerate(all_entries, 1):
        item_idx = item.get("index", idx)
        end_str = str(item.get('end_page')) if item.get('end_page') is not None else "-"
        count_str = str(item.get('page_count')) if item.get('page_count') is not None else "-"
        ch_str = str(item.get('chapter', ''))
        title_str = str(item.get('title', ''))
        start_str = str(item.get('start_page', '-'))
        print(f"{item_idx:<4} {ch_str:<8} {title_str:<28} {start_str:<8} {end_str:<8} {count_str}")
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
