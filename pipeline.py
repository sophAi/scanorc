#!/usr/bin/env python3
"""
scanorc: Scanned Book to Structured HTML / EPUB Pipeline
Using DocLayout-YOLO (Layout Detection & BBox Extraction) + GLM-OCR (Formula, Table & Text Understanding)
Supports:
  1. OCR 區塊批次推論 (--batch-size) 加速單頁內部推論
  2. 多進程並行 (--workers) 加速跨頁全書批次處理 (特別適用於 RTX 5060 Ti 16GB)
"""

import os
import re
import sys
import json
import glob
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
from PIL import Image
from doclayout_yolo import YOLOv10
from transformers import AutoProcessor, GlmOcrForConditionalGeneration

from config_utils import attach_preference_and_template_support

try:
    from recompress_images import recompress_directory
except ImportError:
    recompress_directory = None


def clean_math_syntax(text: str) -> str:
    """
    清理並標準化 LaTeX 數學語法：
    1. 去除行內公式 $ ... $ 內部頭尾的空白字元（Pandoc 與標準 TeX 語法嚴格禁止 $ 緊接空格）。
    2. 將公式末尾包含的句點或逗號移至公式外，例如 `$x.$` -> `$x$.`
    """
    if not text:
        return ""

    def fix_inline_math(match):
        inner = match.group(1).strip()
        trailing_punct = ""
        if inner and inner[-1] in ".，,。；;":
            trailing_punct = inner[-1]
            inner = inner[:-1].strip()
        return f"${inner}${trailing_punct}"

    cleaned = re.sub(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)", fix_inline_math, text)
    return cleaned


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


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def classify_book_image(p: Path) -> int:
    """
    分類書本圖片類型：
    0: 封面/目錄 (Cover, 封面, front_cover 等)
    1: 正文頁面 (Page-xxx, p-xxx, p0xx, 純數字檔名, 頁等)
    2: 其他/未知頁面 (其他插頁等)
    3: 封底/後記 (BackCover, 封底等)
    """
    stem = p.stem.lower()

    # 封底判斷
    if re.search(r"(back[-_]?cover|封底)", stem):
        return 3

    # 封面/目錄判斷
    if re.search(r"(^cover|^封面|^front[-_]?cover)", stem):
        return 0

    # 正文頁碼判斷 (Page-xxx, p-xxx, p0xx, 純數字檔名等)
    if re.search(r"(^page|^p[-_]?\d+|^\d+$|頁|页)", stem):
        return 1

    return 2


def natural_sort_key(p: Path) -> List[Any]:
    """
    檔名自然數字排序鍵 (例如 Cover-05 排在 Cover-06 前面，Page-2 排在 Page-10 前面)
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", p.name)]


def book_image_sort_key(p: Path, cover_first: bool = True) -> Tuple[int, List[Any], str]:
    """
    書本全域排序鍵：
    預設編排順序：Cover-*.jpg (優先) -> Page-*.jpg -> 其他 -> BackCover
    若 cover_first 為 False：Page-*.jpg (優先) -> Cover-*.jpg -> 其他 -> BackCover
    各類別內部依檔名自然數字順序 (natural_sort_key) 排序
    """
    c_type = classify_book_image(p)
    if cover_first:
        priority_map = {0: 0, 1: 1, 2: 2, 3: 3}
    else:
        priority_map = {1: 0, 0: 1, 2: 2, 3: 3}
    priority = priority_map.get(c_type, 2)
    return (priority, natural_sort_key(p), str(p))


def collect_book_images(
    inputs: List[str],
    recursive: bool = True,
    cover_first: bool = True
) -> List[Path]:
    """
    收集並編排所有輸入圖檔：
    1. 支援單一檔案、多個檔案、目錄路徑或萬用字元
    2. 支援遞迴 (recursive) 讀取所有子目錄
    3. 依據 Cover-xx.jpg 與 Page-xxx.jpg 規則排序 (預設 Cover 在前)
    """
    import glob

    collected: List[Path] = []
    seen = set()

    for arg in inputs:
        # 萬用字元支援
        if any(char in arg for char in ["*", "?", "["]):
            matched = glob.glob(arg, recursive=recursive)
            for m in matched:
                mp = Path(m).resolve()
                if mp.is_file() and mp.suffix.lower() in VALID_IMAGE_EXTS and str(mp) not in seen:
                    seen.add(str(mp))
                    collected.append(mp)
            continue

        p = Path(arg).resolve()
        if p.is_file():
            if p.suffix.lower() in VALID_IMAGE_EXTS and str(p) not in seen:
                seen.add(str(p))
                collected.append(p)
            elif p.suffix.lower() not in VALID_IMAGE_EXTS:
                print(f"[-] 略過非支援之圖片檔案: {p.name}", file=sys.stderr)
        elif p.is_dir():
            if recursive:
                for root, _, files in os.walk(p):
                    for f in files:
                        fp = Path(root) / f
                        if fp.suffix.lower() in VALID_IMAGE_EXTS and str(fp) not in seen:
                            seen.add(str(fp))
                            collected.append(fp)
            else:
                for fp in p.iterdir():
                    if fp.is_file() and fp.suffix.lower() in VALID_IMAGE_EXTS and str(fp) not in seen:
                        seen.add(str(fp))
                        collected.append(fp)
        else:
            # 嘗試檢查父目錄的 glob 匹配
            parent = p.parent
            pattern = p.name
            if parent.exists():
                matches = list(parent.glob(pattern))
                for m in matches:
                    if m.is_file() and m.suffix.lower() in VALID_IMAGE_EXTS and str(m) not in seen:
                        seen.add(str(m))
                        collected.append(m)

    # 依照規則排序 (Cover 優先於 Page，各類別內部自然數字排序)
    collected.sort(key=lambda x: book_image_sort_key(x, cover_first=cover_first))
    return collected


def parse_skip_ocr_targets(skip_args: List[str]) -> set:
    """
    解析不作 OCR 的頁面目標清單：
    支援檔名 (Page-050.jpg, page-050)、純頁碼 (50)、純文字清單檔案 (skip.txt)
    傳回標準化之小寫字串集合
    """
    targets = set()
    for item in skip_args:
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
                print(f"[-] 讀取不作 OCR 清單檔案失敗 '{item_str}': {e}", file=sys.stderr)
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


def should_skip_ocr(img_path: Path, skip_targets: set) -> bool:
    """
    判斷特定圖檔是否應跳過 OCR：
    比對完整檔名、stem 或結尾數字頁碼
    """
    if not skip_targets:
        return False
    name_lower = img_path.name.lower()
    stem_lower = img_path.stem.lower()
    if name_lower in skip_targets or stem_lower in skip_targets:
        return True
    nums = re.findall(r"\d+", stem_lower)
    if nums and (nums[-1] in skip_targets or str(int(nums[-1])) in skip_targets):
        return True
    return False


class DocumentConverter:
    def __init__(
        self,
        yolo_weights: str = "models/doclayout_yolo/doclayout_yolo_docstructbench_imgsz1024.pt",
        glm_model_dir: str = "models/glm-ocr",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        max_ocr_dim: int = 1024,
        batch_size: int = 2,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self.max_ocr_dim = max_ocr_dim
        self.batch_size = max(1, batch_size)

        print(f"[*] 載入 DocLayout-YOLO 模型 ({self.device})...")
        self.yolo = YOLOv10(yolo_weights)

        print(f"[*] 載入 GLM-OCR 0.9B 模型與處理器 ({self.device}, {dtype}, batch_size={self.batch_size})...")
        self.processor = AutoProcessor.from_pretrained(glm_model_dir, trust_remote_code=True)
        self.glm = GlmOcrForConditionalGeneration.from_pretrained(
            glm_model_dir,
            dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        self.glm.eval()
        print(f"[+] 所有模型就緒！目前 VRAM 佔用: {round(torch.cuda.memory_allocated() / (1024**2), 1)} MB")

    def run_layout_analysis(self, image_path: str, conf: float = 0.25) -> List[Dict[str, Any]]:
        results = self.yolo.predict(image_path, imgsz=1024, conf=conf, device=self.device)
        boxes_info = []

        im = Image.open(image_path)
        img_w, img_h = im.size

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf_val = float(box.conf[0].item())
                cls_name = self.yolo.names[cls_id]

                xyxy = [max(0, int(x.item())) for x in box.xyxy[0]]
                xyxy[0] = min(img_w, xyxy[0])
                xyxy[1] = min(img_h, xyxy[1])
                xyxy[2] = min(img_w, xyxy[2])
                xyxy[3] = min(img_h, xyxy[3])

                boxes_info.append({
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "conf": conf_val,
                    "bbox": xyxy,
                })

        filtered_boxes = self._filter_contained_boxes(boxes_info)
        sorted_boxes = self._sort_reading_order(filtered_boxes)
        return sorted_boxes

    def _filter_contained_boxes(self, boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = []
        for i, b1 in enumerate(boxes):
            is_contained = False
            for j, b2 in enumerate(boxes):
                if i != j:
                    if self._is_mostly_inside(b1["bbox"], b2["bbox"]) and b1["cls_name"] == b2["cls_name"]:
                        is_contained = True
                        break
            if not is_contained:
                filtered.append(b1)
        return filtered

    def _is_mostly_inside(self, box1: List[int], box2: List[int]) -> bool:
        x1, y1, x2, y2 = box1
        X1, Y1, X2, Y2 = box2
        area1 = (x2 - x1) * (y2 - y1)
        if area1 <= 0:
            return True
        inter_x1 = max(x1, X1)
        inter_y1 = max(y1, Y1)
        inter_x2 = min(x2, X2)
        inter_y2 = min(y2, Y2)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            return (inter_area / area1) > 0.90
        return False

    def _sort_reading_order(self, boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not boxes:
            return []
        return sorted(boxes, key=lambda b: (b["bbox"][1] // 30, b["bbox"][0]))

    def recognize_crops_batch(self, crop_items: List[Tuple[Image.Image, str]]) -> List[str]:
        """
        將多個裁切區塊以 batch_size 分批送入 GLM-OCR 進行平行推論。
        crop_items: List of (PIL.Image, task_type: 'text' | 'formula' | 'table')
        """
        if not crop_items:
            return []

        task_prompts = {
            "text": "Text Recognition:",
            "formula": "Formula Recognition:",
            "table": "Table Recognition:",
        }

        all_results = []
        total_items = len(crop_items)

        for i in range(0, total_items, self.batch_size):
            chunk = crop_items[i : i + self.batch_size]

            batch_imgs = []
            batch_prompts = []
            has_table = False

            for crop_img, task_type in chunk:
                if task_type == "table":
                    has_table = True
                prompt_text = task_prompts.get(task_type, "Text Recognition:")

                w, h = crop_img.size
                ocr_img = crop_img.copy()
                if max(w, h) > self.max_ocr_dim:
                    ocr_img.thumbnail((self.max_ocr_dim, self.max_ocr_dim), Image.Resampling.LANCZOS)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ]
                p = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                batch_imgs.append(ocr_img)
                batch_prompts.append(p)

            inputs = self.processor(
                images=batch_imgs,
                text=batch_prompts,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            max_tokens = 1024 if has_table else 512
            with torch.no_grad():
                outputs = self.glm.generate(**inputs, max_new_tokens=max_tokens)

            # 解碼各批次結果
            for out_seq in outputs:
                text_res = self.processor.decode(out_seq[inputs.input_ids.shape[1]:], skip_special_tokens=True)
                all_results.append(clean_math_syntax(text_res.strip()))

        return all_results

    def process_page(
        self,
        image_path: str,
        output_dir: str,
        page_id: str = None,
        skip_ocr: bool = False
    ) -> Dict[str, Any]:
        """
        完整處理單一頁面：
        1. 若指定 skip_ocr=True，直接以最高畫質整頁原始圖檔嵌入，跳過版面分析與 OCR 推論
        2. 版面分析與元素分類 (DocLayout-YOLO)
        3. 插圖原圖最高解析度自動裁切
        4. 聚合文字/公式/表格區塊進行 Batch OCR 推論 (GLM-OCR)
        5. 若 OCR 過程發生非預期失敗，自動優雅降級為全頁原始圖檔嵌入
        """
        if page_id is None:
            page_id = Path(image_path).name
        page_stem = Path(image_path).stem

        images_out_dir = os.path.join(output_dir, "images")
        os.makedirs(images_out_dir, exist_ok=True)

        # 1. 主動跳過 OCR 模式 (使用者指定全頁圖片頁或免 OCR 頁)
        if skip_ocr:
            print(f"\n==================================================")
            print(f"[*] 處理頁面: {page_id} (指定不作 OCR / 整頁插圖模式)")
            print(f"==================================================")
            dest_img_filename = f"{page_stem}_raw{Path(image_path).suffix.lower()}"
            dest_img_path = os.path.join(images_out_dir, dest_img_filename)
            shutil.copy2(image_path, dest_img_path)
            print(f"  [+] 成功將原始圖檔嵌入為整頁插圖: images/{dest_img_filename}")
            return {
                "page_id": page_id,
                "page_stem": page_stem,
                "blocks": [{
                    "type": "figure",
                    "img_src": f"images/{dest_img_filename}",
                    "caption": "",
                    "is_full_page": True
                }]
            }

        orig_im = Image.open(image_path).convert("RGB")
        img_w, img_h = orig_im.size

        print(f"\n==================================================")
        print(f"[*] 處理頁面: {page_id} ({img_w}x{img_h})")
        print(f"==================================================")

        try:
            boxes = self.run_layout_analysis(image_path)

            # 整理區塊資訊
            items_to_ocr = []  # List of (crop_img, task_type, block_meta)
            html_blocks_draft = []
            fig_count = 0

            for idx, box_info in enumerate(boxes):
                cls_name = box_info["cls_name"]
                bbox = box_info["bbox"]
                conf = box_info["conf"]
                x1, y1, x2, y2 = bbox

                if (x2 - x1) < 15 or (y2 - y1) < 15:
                    continue

                # 捨棄頁眉、頁腳、頁碼
                if cls_name == "abandon":
                    print(f"  [-] [拋棄] {cls_name:15s} (conf: {conf:.2f}) bbox: {bbox}")
                    continue

                pad = 2
                crop_box = (
                    max(0, x1 - pad),
                    max(0, y1 - pad),
                    min(img_w, x2 + pad),
                    min(img_h, y2 + pad)
                )
                crop_im = orig_im.crop(crop_box)

                # 插圖 (Figure) 無需 OCR，直接儲存原圖
                if cls_name == "figure":
                    fig_count += 1
                    fig_filename = f"{page_stem}_fig_{fig_count:02d}.png"
                    fig_path = os.path.join(images_out_dir, fig_filename)
                    crop_im.save(fig_path, "PNG", quality=95)
                    print(f"  [+] [插圖] 成功裁切原始高解析圖檔: images/{fig_filename} (尺寸: {crop_im.size})")
                    html_blocks_draft.append({
                        "type": "figure",
                        "img_src": f"images/{fig_filename}",
                        "caption": "",
                        "is_full_page": False
                    })

                elif cls_name == "figure_caption":
                    items_to_ocr.append((crop_im, "text", {"type": "figure_caption", "draft_idx": len(html_blocks_draft)}))
                    html_blocks_draft.append({"type": "caption", "text": ""})

                elif cls_name == "table":
                    items_to_ocr.append((crop_im, "table", {"type": "table", "draft_idx": len(html_blocks_draft)}))
                    html_blocks_draft.append({"type": "table", "content": ""})

                elif cls_name == "isolate_formula":
                    items_to_ocr.append((crop_im, "formula", {"type": "formula", "draft_idx": len(html_blocks_draft)}))
                    html_blocks_draft.append({"type": "formula", "content": ""})

                elif cls_name == "title":
                    items_to_ocr.append((crop_im, "text", {"type": "title", "draft_idx": len(html_blocks_draft)}))
                    html_blocks_draft.append({"type": "title", "content": ""})

                elif cls_name == "plain text":
                    items_to_ocr.append((crop_im, "text", {"type": "text", "draft_idx": len(html_blocks_draft)}))
                    html_blocks_draft.append({"type": "text", "content": ""})

                else:
                    items_to_ocr.append((crop_im, "text", {"type": "text", "draft_idx": len(html_blocks_draft)}))
                    html_blocks_draft.append({"type": "text", "content": ""})

            # 批次 OCR 推論
            if items_to_ocr:
                print(f"  [*] 進行 Batch OCR 辨識 (共 {len(items_to_ocr)} 個區塊, batch_size={self.batch_size})...")
                crops_and_types = [(item[0], item[1]) for item in items_to_ocr]
                ocr_results = self.recognize_crops_batch(crops_and_types)

                # 將結果填回各區塊草稿
                for (_, _, meta), res_text in zip(items_to_ocr, ocr_results):
                    draft_idx = meta["draft_idx"]
                    b_type = meta["type"]

                    if b_type == "figure_caption":
                        print(f"  [+] [圖說] {res_text}")
                        # 若前一個是 figure 則綁定
                        if draft_idx > 0 and html_blocks_draft[draft_idx - 1]["type"] == "figure":
                            html_blocks_draft[draft_idx - 1]["caption"] = res_text
                            html_blocks_draft[draft_idx] = None  # 標記移除
                        else:
                            html_blocks_draft[draft_idx]["text"] = res_text

                    elif b_type == "table":
                        print(f"  [+] [表格完成] 長度 {len(res_text)} 字元")
                        html_blocks_draft[draft_idx]["content"] = res_text

                    elif b_type == "formula":
                        print(f"  [+] [公式完成] {res_text[:50]}...")
                        if not res_text.startswith("$$"):
                            res_text = f"$$\n{res_text}\n$$"
                        html_blocks_draft[draft_idx]["content"] = res_text

                    elif b_type == "title":
                        print(f"  [+] [標題] {res_text}")
                        html_blocks_draft[draft_idx]["content"] = res_text

                    else:
                        print(f"  [+] [內文] {res_text[:35]}...")
                        html_blocks_draft[draft_idx]["content"] = res_text

            # 過濾已合併的區塊
            final_blocks = [b for b in html_blocks_draft if b is not None]

            # 若此頁完全未識別出任何有效內容（例如空頁或純雜訊），自動轉為原圖嵌入
            if not final_blocks:
                print(f"  [-] 該頁未偵測到任何內文區塊，自動將原圖嵌入為整頁插圖")
                dest_img_filename = f"{page_stem}_raw{Path(image_path).suffix.lower()}"
                dest_img_path = os.path.join(images_out_dir, dest_img_filename)
                shutil.copy2(image_path, dest_img_path)
                final_blocks = [{
                    "type": "figure",
                    "img_src": f"images/{dest_img_filename}",
                    "caption": "",
                    "is_full_page": True
                }]

            return {
                "page_id": page_id,
                "page_stem": page_stem,
                "blocks": final_blocks
            }

        except Exception as e:
            print(f"  [-] 警告：頁面 '{page_id}' OCR 辨識失敗或難度過高 ({e})，自動降級為整頁原始圖檔嵌入！", file=sys.stderr)
            dest_img_filename = f"{page_stem}_raw{Path(image_path).suffix.lower()}"
            dest_img_path = os.path.join(images_out_dir, dest_img_filename)
            shutil.copy2(image_path, dest_img_path)
            return {
                "page_id": page_id,
                "page_stem": page_stem,
                "blocks": [{
                    "type": "figure",
                    "img_src": f"images/{dest_img_filename}",
                    "caption": "",
                    "is_full_page": True
                }]
            }

    def render_html(
        self,
        page_results: List[Dict[str, Any]],
        title: str = "Scanned Book",
        toc_data: Optional[Dict[str, Any]] = None
    ) -> str:
        body_content = []

        available_page_files = [p["page_id"] for p in page_results]

        # 建立章節對應至起始頁面的映射 (EPUB outline 核心)
        page_chapter_map: Dict[str, List[Dict[str, Any]]] = {}
        if toc_data and "toc" in toc_data and toc_data["toc"]:
            toc_entries = toc_data["toc"]
            page_numbers: List[Tuple[Optional[int], str]] = []
            for p_file in available_page_files:
                nums = re.findall(r"\d+", p_file)
                p_num = int(nums[-1]) if nums else None
                page_numbers.append((p_num, p_file))

            for c_idx, entry in enumerate(toc_entries, start=1):
                start_p = entry.get("start_page", 0)
                end_p = entry.get("end_page", start_p)
                target_file = None
                # 優先 1: 精確比對頁碼
                for p_num, p_file in page_numbers:
                    if p_num is not None and p_num == start_p:
                        target_file = p_file
                        break
                # 優先 2: 落在章節頁碼範圍內的第一個頁面
                if not target_file:
                    for p_num, p_file in page_numbers:
                        if p_num is not None and start_p <= p_num <= end_p:
                            target_file = p_file
                            break
                # 優先 3: 依檔名規則推算 fallback
                if not target_file:
                    target_file = resolve_page_filename(start_p, available_page_files)

                entry_info = dict(entry)
                entry_info["_chapter_index"] = c_idx
                if target_file not in page_chapter_map:
                    page_chapter_map[target_file] = []
                page_chapter_map[target_file].append(entry_info)

        # 插入目錄索引導覽區塊 (TOC)
        if toc_data and "toc" in toc_data and toc_data["toc"]:
            toc_entries = toc_data["toc"]
            toc_title = toc_data.get("book_title", title)

            body_content.append('<h1 id="toc-heading" class="chapter-heading">目錄</h1>')
            body_content.append('<nav id="toc" class="book-toc" role="doc-toc">')
            body_content.append(f'  <h2 class="toc-title">目錄 ({toc_title})</h2>')
            body_content.append('  <ul class="toc-list">')

            for c_idx, entry in enumerate(toc_entries, start=1):
                start_p = entry.get("start_page", 0)
                full_t = entry.get("full_title", entry.get("title", ""))
                target_file = resolve_page_filename(start_p, available_page_files)

                body_content.append('    <li class="toc-item">')
                body_content.append(f'      <a href="#chapter-{c_idx}" class="toc-link">')
                body_content.append(f'        <span class="toc-chapter">{full_t}</span>')
                body_content.append('        <span class="toc-dots"></span>')
                body_content.append(f'        <span class="toc-page">第 {start_p} 頁 ({target_file})</span>')
                body_content.append('      </a>')
                body_content.append('    </li>')

            body_content.append('  </ul>')
            body_content.append('</nav>\n<hr class="page-divider">\n')

        for p_res in page_results:
            page_id = p_res["page_id"]
            page_stem = p_res.get("page_stem", Path(page_id).stem)

            # 若此頁為章節起始頁，先插入 <h1> 章節標題 (Pandoc EPUB Outline 自動識別)
            if page_id in page_chapter_map:
                for chap in page_chapter_map[page_id]:
                    chap_idx = chap.get("_chapter_index", 1)
                    full_t = chap.get("full_title", chap.get("title", f"第 {chap_idx} 章"))
                    body_content.append(f'<h1 id="chapter-{chap_idx}" class="chapter-heading">{full_t}</h1>\n')

            body_content.append(f'<section class="book-page" id="{page_id}">')
            body_content.append(f'  <a id="{page_stem}" class="page-anchor-target"></a>')
            body_content.append(f'  <div class="page-anchor"><small>File: {page_id}</small></div>')

            for b in p_res["blocks"]:
                b_type = b["type"]
                if b_type == "title":
                    body_content.append(f'  <h2 class="book-title">{b["content"]}</h2>')
                elif b_type == "text":
                    paragraphs = b["content"].split("\n")
                    for p in paragraphs:
                        p = p.strip()
                        if p:
                            body_content.append(f'  <p>{p}</p>')
                elif b_type == "figure":
                    cap = b.get("caption", "")
                    is_full = b.get("is_full_page", False)
                    css_class = "book-figure fullpage-figure" if is_full else "book-figure"
                    fig_html = f'  <figure class="{css_class}">\n    <img src="{b["img_src"]}" alt="{cap}">\n'
                    if cap and not is_full:
                        fig_html += f'    <figcaption>{cap}</figcaption>\n'
                    fig_html += '  </figure>'
                    body_content.append(fig_html)
                elif b_type == "caption":
                    body_content.append(f'  <p class="book-caption">{b["text"]}</p>')
                elif b_type == "table":
                    body_content.append(f'  <div class="table-container">\n    {b["content"]}\n  </div>')
                elif b_type == "formula":
                    body_content.append(f'  <div class="math-block">\n    {b["content"]}\n  </div>')

            body_content.append('</section>\n<hr class="page-divider">\n')

        body_html_str = "\n".join(body_content)

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
      color: #999;
      margin-bottom: 8px;
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
  </style>
</head>
<body>
{body_html_str}
</body>
</html>
"""
        return full_html


def build_epub_with_pandoc(
    html_path: str,
    epub_path: str,
    title: str = "Scanned Book",
    cover_image: Optional[str] = None,
    split_level: int = 1,
):
    pandoc_bin = "/opt/anaconda3/bin/pandoc"
    if not os.path.exists(pandoc_bin):
        pandoc_bin = "pandoc"

    html_dir = os.path.dirname(os.path.abspath(html_path))
    html_file = os.path.basename(html_path)
    abs_epub_path = os.path.abspath(epub_path)

    cmd = [
        pandoc_bin,
        "-f", "html+tex_math_dollars",
        html_file,
        "-o", abs_epub_path,
        "--resource-path=.",
        "--mathml",
        "--toc",
        "--toc-depth=3",
        f"--split-level={split_level}",
        "--metadata", f"title={title}",
        "--metadata", "language=zh-TW",
    ]
    if cover_image and os.path.exists(cover_image):
        cmd.append(f"--epub-cover-image={os.path.abspath(cover_image)}")

    print(f"[*] 執行 Pandoc 編譯 MathML EPUB (工作目錄: {html_dir}):\n    {' '.join(cmd)}")
    subprocess.run(cmd, cwd=html_dir, check=True)
    print(f"[+] EPUB 生成成功 (含原生 MathML 渲染與章節 Outline): {abs_epub_path}")


def _worker_process_page(args_tuple):
    """多進程 Worker 任務函數"""
    task_idx, img_path, output_dir, batch_size, skip_ocr = args_tuple
    # 每個子進程獨立初始化 Converter
    converter = DocumentConverter(batch_size=batch_size)
    res = converter.process_page(img_path, output_dir, skip_ocr=skip_ocr)
    res["_task_idx"] = task_idx
    return res


def main():
    parser = argparse.ArgumentParser(
        description="ScanORC: Scanned Document to HTML / EPUB Pipeline (支援遞迴子目錄與 Cover-Page 智能排序)"
    )
    parser.add_argument(
        "--samples", "-i", "--input",
        nargs="+",
        default=["samples"],
        dest="samples",
        help="待處理圖檔目錄或圖片路徑 (支援單檔、多檔、目錄或萬用字元，預設: samples)"
    )
    parser.add_argument("--output", default="output", help="輸出目錄 (預設: output)")
    parser.add_argument("--title", default="數位化掃描書本", help="書名 / 文檔標題")
    parser.add_argument("--toc", default=None, help="目錄結構 JSON 檔案路徑 (例如 toc.json，將自動生成目錄索引並連結至各頁檔名)")
    parser.add_argument("--epub", action="store_true", help="是否自動生成 EPUB")
    parser.add_argument(
        "--cover", "--cover-image",
        default=None,
        dest="cover",
        help="指定 EPUB 封面圖片檔案路徑 (若未指定將自動在待處理圖片中尋找 Cover-*.jpg)"
    )
    parser.add_argument(
        "--compress-images",
        action="store_true",
        help="在編譯 EPUB 前，自動對 output/images 目錄中的 JPG 與 PNG 圖檔進行重新壓縮與尺寸縮放"
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
    parser.add_argument("--batch-size", type=int, default=2, help="單頁 OCR 區塊批次大小 (預設 2，16GB 顯卡可設 4~8)")
    parser.add_argument("--workers", type=int, default=1, help="多進程並行 Worker 數 (預設 1，16GB 顯卡可設 3~4)")
    parser.add_argument(
        "--recursive", "-r",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否遞迴讀取子目錄所有圖檔 (預設: True，使用 --no-recursive 僅讀取頂層目錄)"
    )
    parser.add_argument(
        "--cover-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否將封面/目錄頁 (Cover-*.jpg) 編排於內文頁 (Page-*.jpg) 之前 (預設: True，使用 --no-cover-first 則內文頁優先)"
    )
    parser.add_argument(
        "--skip-ocr", "--skip-orc", "--no-ocr",
        nargs="*",
        default=[],
        dest="skip_ocr",
        help="不作 OCR 的檔案清單 (可指定檔名、頁碼或包含清單的純文字檔路徑，直接以原始圖檔嵌入 HTML/EPUB)"
    )
    args = attach_preference_and_template_support(parser, "pipeline")

    os.makedirs(args.output, exist_ok=True)

    sample_inputs = [args.samples] if isinstance(args.samples, str) else list(args.samples)

    # 載入目錄結構資料 (若有指定 --toc 或自動在當前目錄、samples、output 找到 toc.json)
    toc_data = None
    toc_file = args.toc
    if not toc_file:
        candidates = [Path("toc.json"), Path(args.output) / "toc.json"]
        for s_arg in sample_inputs:
            sp = Path(s_arg)
            if sp.is_dir():
                candidates.append(sp / "toc.json")
            elif sp.is_file():
                candidates.append(sp.parent / "toc.json")
        for candidate in candidates:
            if candidate.exists():
                toc_file = str(candidate)
                break

    if toc_file and os.path.exists(toc_file):
        try:
            with open(toc_file, "r", encoding="utf-8") as f:
                toc_data = json.load(f)
            total_ch = len(toc_data.get("toc", []))
            print(f"[*] 載入目錄索引檔案: {toc_file} (包含 {total_ch} 個章節索引)")
            # 若使用者未特別自訂書名，自動套用 toc.json 中的書名
            if args.title == "數位化掃描書本" and toc_data.get("book_title"):
                args.title = toc_data["book_title"]
        except Exception as e:
            print(f"[-] 讀取目錄索引檔案失敗: {e}", file=sys.stderr)

    image_files = collect_book_images(
        inputs=sample_inputs,
        recursive=args.recursive,
        cover_first=args.cover_first
    )

    if not image_files:
        print(f"[-] 未找到待處理之圖片檔案！(搜尋路徑: {sample_inputs})", file=sys.stderr)
        sys.exit(1)

    skip_targets = parse_skip_ocr_targets(args.skip_ocr)
    if skip_targets:
        print(f"[*] 設定不作 OCR 頁面清單 (共 {len(skip_targets)} 個目標比對標記)")

    cover_imgs = [p for p in image_files if classify_book_image(p) == 0]
    page_imgs = [p for p in image_files if classify_book_image(p) == 1]
    other_imgs = [p for p in image_files if classify_book_image(p) not in (0, 1)]

    print(f"[*] 找到 {len(image_files)} 個頁面準備處理 (遞迴={args.recursive}, 封面優先={args.cover_first})：")
    if cover_imgs:
        print(f"    - 封面/目錄頁: {len(cover_imgs)} 頁 {[p.name for p in cover_imgs[:6]]}{' ...' if len(cover_imgs) > 6 else ''}")
    if page_imgs:
        print(f"    - 正文頁面:   {len(page_imgs)} 頁 (起始: {page_imgs[0].name}, 結束: {page_imgs[-1].name})")
    if other_imgs:
        print(f"    - 其他/插頁:   {len(other_imgs)} 頁 {[p.name for p in other_imgs[:6]]}{' ...' if len(other_imgs) > 6 else ''}")
    print(f"[*] 頁面編排順序前 10 頁: {[p.name for p in image_files[:10]]}")
    print(f"[*] 運算設定: batch_size={args.batch_size}, workers={args.workers}, 書名='{args.title}'")

    all_page_results = []

    if args.workers > 1:
        print(f"[*] 啟動多進程並行處理 (Worker 數: {args.workers})...")
        import torch.multiprocessing as mp
        # 設置 spawn 模式以安全共享 CUDA
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        tasks = [
            (idx, str(img_path), args.output, args.batch_size, should_skip_ocr(img_path, skip_targets))
            for idx, img_path in enumerate(image_files)
        ]
        with mp.Pool(processes=args.workers) as pool:
            all_page_results = pool.map(_worker_process_page, tasks)
        # 確保結果嚴格依照任務與編排順序排序
        all_page_results.sort(key=lambda x: x["_task_idx"])
    else:
        converter = DocumentConverter(batch_size=args.batch_size)
        for idx, img_path in enumerate(image_files):
            p_res = converter.process_page(
                str(img_path),
                args.output,
                skip_ocr=should_skip_ocr(img_path, skip_targets)
            )
            p_res["_task_idx"] = idx
            all_page_results.append(p_res)

    html_out_path = os.path.join(args.output, "book.html")
    # 初始化一個臨時實例來 render HTML
    temp_conv = DocumentConverter.__new__(DocumentConverter)
    full_html = temp_conv.render_html(all_page_results, title=args.title, toc_data=toc_data)
    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"\n[+] 完整 HTML 檔案生成完畢 (含 TOC 目錄導覽索引與章節 Outline): {html_out_path}")

    # 執行 images 目錄重新壓縮與縮放 (若啟用)
    images_dir = Path(args.output) / "images"
    if args.compress_images and images_dir.exists() and any(images_dir.iterdir()):
        if recompress_directory is None:
            print("[-] 警告: 未能載入 recompress_images 模組，略過圖片壓縮步驟", file=sys.stderr)
        else:
            print(f"[*] 正在對 {images_dir} 進行 JPG/PNG 圖片壓縮與縮放 (品質: {args.image_quality}, 縮放: {args.image_scale}, 最大尺寸: {args.image_max_dim})...")
            tmp_opt_dir = Path(args.output) / "images_opt_tmp"
            if tmp_opt_dir.exists():
                shutil.rmtree(tmp_opt_dir)
            try:
                recompress_directory(
                    input_dir=images_dir,
                    output_dir=tmp_opt_dir,
                    qualities=[args.image_quality],
                    scales=[args.image_scale],
                    max_dim=args.image_max_dim,
                    workers=min(os.cpu_count() or 4, 16),
                    verbose=True,
                )
                shutil.rmtree(images_dir)
                tmp_opt_dir.rename(images_dir)
                print(f"[+] images 目錄圖檔重新壓縮與縮放完成！")
            except Exception as e:
                print(f"[-] 圖片壓縮過程異常: {e}", file=sys.stderr)
                if tmp_opt_dir.exists():
                    shutil.rmtree(tmp_opt_dir)

    if args.epub:
        # 偵測或套用封面圖片
        actual_cover = args.cover
        if not actual_cover:
            if cover_imgs:
                actual_cover = str(cover_imgs[0])
            else:
                for cand in [Path(args.output) / "cover.jpg", Path(args.output) / "Cover.jpg"]:
                    if cand.exists():
                        actual_cover = str(cand)
                        break

        if actual_cover and os.path.exists(actual_cover):
            print(f"[*] 套用 EPUB 封面圖檔: {actual_cover}")

        epub_out_path = os.path.join(args.output, "book.epub")
        build_epub_with_pandoc(
            html_out_path,
            epub_out_path,
            title=args.title,
            cover_image=actual_cover,
            split_level=1,
        )

    print("\n[🎉] 全流程轉換成功完成！")


if __name__ == "__main__":
    main()
