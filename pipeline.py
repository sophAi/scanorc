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
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
from PIL import Image
from doclayout_yolo import YOLOv10
from transformers import AutoProcessor, GlmOcrForConditionalGeneration


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
        page_id: str = None
    ) -> Dict[str, Any]:
        """
        完整處理單一頁面：
        1. 版面分析與元素分類
        2. 插圖原圖最高解析度自動裁切
        3. 聚合文字/公式/表格區塊進行 Batch OCR 推論
        """
        if page_id is None:
            page_id = Path(image_path).name
        page_stem = Path(image_path).stem

        images_out_dir = os.path.join(output_dir, "images")
        os.makedirs(images_out_dir, exist_ok=True)

        orig_im = Image.open(image_path).convert("RGB")
        img_w, img_h = orig_im.size

        print(f"\n==================================================")
        print(f"[*] 處理頁面: {page_id} ({img_w}x{img_h})")
        print(f"==================================================")

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
                    "caption": ""
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

        return {
            "page_id": page_id,
            "page_stem": page_stem,
            "blocks": final_blocks
        }

    def render_html(
        self,
        page_results: List[Dict[str, Any]],
        title: str = "Scanned Book",
        toc_data: Optional[Dict[str, Any]] = None
    ) -> str:
        body_content = []

        available_page_files = [p["page_id"] for p in page_results]

        # 插入目錄索引導覽區塊 (TOC)
        if toc_data and "toc" in toc_data and toc_data["toc"]:
            toc_entries = toc_data["toc"]
            toc_title = toc_data.get("book_title", title)

            body_content.append('<nav id="toc" class="book-toc" role="doc-toc">')
            body_content.append(f'  <h2 class="toc-title">目錄 ({toc_title})</h2>')
            body_content.append('  <ul class="toc-list">')

            for entry in toc_entries:
                start_p = entry.get("start_page", 0)
                full_t = entry.get("full_title", entry.get("title", ""))
                # 自動對應至檔案名稱 (如 Page-050.jpg)
                target_file = resolve_page_filename(start_p, available_page_files)

                body_content.append('    <li class="toc-item">')
                body_content.append(f'      <a href="#{target_file}" class="toc-link">')
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
                    fig_html = f'  <figure class="book-figure">\n    <img src="{b["img_src"]}" alt="{cap}">\n'
                    if cap:
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


def build_epub_with_pandoc(html_path: str, epub_path: str, title: str = "Scanned Book"):
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
        "--metadata", f"title={title}",
        "--metadata", "language=zh-TW",
    ]
    print(f"[*] 執行 Pandoc 編譯 MathML EPUB (工作目錄: {html_dir}):\n    {' '.join(cmd)}")
    subprocess.run(cmd, cwd=html_dir, check=True)
    print(f"[+] EPUB 生成成功 (含原生 MathML 渲染): {abs_epub_path}")


def _worker_process_page(args_tuple):
    """多進程 Worker 任務函數"""
    task_idx, img_path, output_dir, batch_size = args_tuple
    # 每個子進程獨立初始化 Converter
    converter = DocumentConverter(batch_size=batch_size)
    res = converter.process_page(img_path, output_dir)
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
    args = parser.parse_args()

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

        tasks = [(idx, str(img_path), args.output, args.batch_size) for idx, img_path in enumerate(image_files)]
        with mp.Pool(processes=args.workers) as pool:
            all_page_results = pool.map(_worker_process_page, tasks)
        # 確保結果嚴格依照任務與編排順序排序
        all_page_results.sort(key=lambda x: x["_task_idx"])
    else:
        converter = DocumentConverter(batch_size=args.batch_size)
        for idx, img_path in enumerate(image_files):
            p_res = converter.process_page(str(img_path), args.output)
            p_res["_task_idx"] = idx
            all_page_results.append(p_res)

    html_out_path = os.path.join(args.output, "book.html")
    # 初始化一個臨時實例來 render HTML
    temp_conv = DocumentConverter.__new__(DocumentConverter)
    full_html = temp_conv.render_html(all_page_results, title=args.title, toc_data=toc_data)
    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"\n[+] 完整 HTML 檔案生成完畢 (含 TOC 目錄導覽索引): {html_out_path}")

    if args.epub:
        epub_out_path = os.path.join(args.output, "book.epub")
        build_epub_with_pandoc(html_out_path, epub_out_path, title=args.title)

    print("\n[🎉] 全流程轉換成功完成！")


if __name__ == "__main__":
    main()
