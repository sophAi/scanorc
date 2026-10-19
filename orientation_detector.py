#!/usr/bin/env python3
"""
ScanORC: 頁面與書籍排版方向自動偵測工具 (orientation_detector.py)

功能：
  1. 獨立命令列工具與 Python API，輸出 'vertical' 或 'horizon' (或 'horizontal') 字串。
  2. 支援標準 ScanORC JSON 偏好設定介面 (-p / -c / -t / 單一 JSON 直通)。
  3. 優先以 Page-xxx.jpg 正文圖檔進行抽樣，自動排除 Cover-* 與 Content-* 圖檔。
  4. 結合影像水平/垂直投影方差比 (Projection Profile Variance, PPV) 與 YOLO BBox 長寬比。
  5. 支援 --quiet 精簡模式，供 batch_process.py 或 process_book.py 子進程直接調用。
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Set

import cv2
import numpy as np
from PIL import Image

from config_utils import attach_preference_and_template_support

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def normalize_layout_direction(val: Optional[str]) -> str:
    """
    將排版方向值正規化為標準的 'vertical' 或 'horizontal'，若未提供或無法辨識則回傳 'auto'。
    支援別名: vertical, vert, vertical-rl, v -> vertical
              horizontal, horizon, horizontal-tb, h -> horizontal
    """
    if not val:
        return "auto"
    s = str(val).strip().lower()
    if s in ("vertical", "vert", "vertical-rl", "v"):
        return "vertical"
    if s in ("horizontal", "horizon", "horizontal-tb", "h"):
        return "horizontal"
    return "auto"


def natural_sort_key(s: Any) -> List:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def detect_orientation_projection(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    sample_width: int = 800,
    threshold_vertical_ratio: float = 1.35,
    threshold_horizontal_ratio: float = 0.75,
) -> Dict[str, Any]:
    """
    透過水平與垂直投影輪廓之像素分佈變異數 (Variance)，判定頁面為直排或橫排。

    原理：
      - 橫排版面：水平方向具有交替之文字行與行距空白，水平投影波動極大 (Var_H 高)，垂直投影相對平緩。
      - 直排版面：垂直方向具有交替之文字欄與欄距空白，垂直投影波動極大 (Var_V 高)，水平投影相對平緩。
      - Ratio = Var_V / Var_H：Ratio > 1.35 傾向直排；Ratio < 0.75 傾向橫排。
    """
    if isinstance(image_input, (str, Path)):
        img_path_str = str(image_input)
        img = cv2.imread(img_path_str, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {
                "orientation": "horizontal",
                "confidence": 0.0,
                "ratio": 1.0,
                "error": f"無法讀取圖片: {img_path_str}"
            }
    elif isinstance(image_input, Image.Image):
        img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2GRAY)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            img = image_input
    else:
        raise ValueError(f"不支援的圖片輸入類型: {type(image_input)}")

    h, w = img.shape
    if w <= 0 or h <= 0:
        return {"orientation": "horizontal", "confidence": 0.0, "ratio": 1.0}

    # 縮圖至固定寬度 sample_width (如 800px) 以達成 3~8ms 之極速運算
    if w > sample_width:
        scale = sample_width / float(w)
        resized_h = max(10, int(h * scale))
        resized = cv2.resize(img, (sample_width, resized_h), interpolation=cv2.INTER_AREA)
    else:
        resized = img

    # Otsu 二值化反轉 (前景文字為 255，背景留白為 0)
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 計算投影 (每列/每欄的前景像素累計值)
    proj_h = np.sum(binary, axis=1)  # 水平投影 (長度為高度)
    proj_v = np.sum(binary, axis=0)  # 垂直投影 (長度為寬度)

    # 過濾留白邊界 (僅取高於均值 5% 之有效正文區域)
    mean_h = np.mean(proj_h)
    mean_v = np.mean(proj_v)
    valid_h = proj_h[proj_h > mean_h * 0.05]
    valid_v = proj_v[proj_v > mean_v * 0.05]

    var_h = float(np.var(valid_h)) if len(valid_h) > 5 else 1.0
    var_v = float(np.var(valid_v)) if len(valid_v) > 5 else 1.0

    ratio = var_v / max(var_h, 1e-5)

    if ratio >= threshold_vertical_ratio:
        orientation = "vertical"
        confidence = min(1.0, max(0.6, (ratio - 1.0) / 4.0 + 0.6))
    elif ratio <= threshold_horizontal_ratio:
        orientation = "horizontal"
        confidence = min(1.0, max(0.6, (1.0 - ratio) * 0.8 + 0.6))
    else:
        orientation = "ambiguous"
        confidence = 0.5

    return {
        "orientation": orientation,
        "confidence": round(confidence, 3),
        "ratio": round(ratio, 4),
        "var_h": round(var_h, 1),
        "var_v": round(var_v, 1),
    }


def detect_orientation_bboxes(
    boxes: List[Dict[str, Any]],
    text_classes: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    透過 DocLayout-YOLO 預測之文字 bounding box 長寬比統計，輔助判定頁面排版方向。
    """
    if text_classes is None:
        text_classes = {"plain text", "title", "text", "table_footnote"}

    text_boxes = [b for b in boxes if b.get("cls_name") in text_classes]
    if not text_boxes:
        return {
            "orientation": "ambiguous",
            "confidence": 0.0,
            "v_count": 0,
            "h_count": 0,
            "mean_aspect_ratio": 1.0,
        }

    aspect_ratios = []
    v_weight = 0.0
    h_weight = 0.0
    v_count = 0
    h_count = 0

    for b in text_boxes:
        xyxy = b.get("bbox", [0, 0, 0, 0])
        w = max(1, xyxy[2] - xyxy[0])
        h = max(1, xyxy[3] - xyxy[1])
        aspect = w / float(h)
        area = w * h
        aspect_ratios.append(aspect)

        # 寬度顯著小於高度 (直長條 column)
        if aspect < 0.75:
            v_weight += area
            v_count += 1
        # 寬度顯著大於高度 (橫長條 line)
        elif aspect > 1.3:
            h_weight += area
            h_count += 1

    total_weight = v_weight + h_weight
    total_count = v_count + h_count

    if total_weight <= 0 or total_count == 0:
        return {
            "orientation": "ambiguous",
            "confidence": 0.0,
            "v_count": v_count,
            "h_count": h_count,
            "mean_aspect_ratio": round(float(np.mean(aspect_ratios)), 3) if aspect_ratios else 1.0,
        }

    v_ratio = v_weight / total_weight
    mean_aspect = float(np.mean(aspect_ratios))

    if v_ratio > 0.6:
        orientation = "vertical"
        confidence = round(v_ratio, 3)
    elif v_ratio < 0.4:
        orientation = "horizontal"
        confidence = round(1.0 - v_ratio, 3)
    else:
        orientation = "ambiguous"
        confidence = 0.5

    return {
        "orientation": orientation,
        "confidence": confidence,
        "v_count": v_count,
        "h_count": h_count,
        "v_weight_ratio": round(v_ratio, 3),
        "mean_aspect_ratio": round(mean_aspect, 3),
    }


def detect_page_orientation(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    boxes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    結合投影方差分析與 YOLO BBox 統計的綜合單頁判定。
    """
    proj_res = detect_orientation_projection(image_input)

    if boxes is not None and len(boxes) > 0:
        bbox_res = detect_orientation_bboxes(boxes)
        if proj_res["orientation"] == bbox_res["orientation"] and proj_res["orientation"] != "ambiguous":
            return {
                "orientation": proj_res["orientation"],
                "confidence": min(1.0, max(proj_res["confidence"], bbox_res["confidence"]) + 0.1),
                "method": "hybrid_consensus",
                "projection": proj_res,
                "bbox": bbox_res,
            }
        if proj_res["orientation"] == "ambiguous" and bbox_res["orientation"] != "ambiguous":
            return {
                "orientation": bbox_res["orientation"],
                "confidence": bbox_res["confidence"],
                "method": "bbox_primary",
                "projection": proj_res,
                "bbox": bbox_res,
            }
        # 當 BBox 統計具有明確排版方向 (例如文字框皆為橫向或豎向，且反方向計數為 0 或置信度 >= 0.80)
        # 且投影法置信度未達絕對優勢 (如 < 0.85 或低於 BBox 置信度) 時，以文字框排版方向為準
        if bbox_res["orientation"] != "ambiguous":
            has_exclusive_direction = (bbox_res["v_count"] == 0 and bbox_res["h_count"] >= 2) or \
                                      (bbox_res["h_count"] == 0 and bbox_res["v_count"] >= 2)
            if (has_exclusive_direction and bbox_res["confidence"] >= 0.75) or \
               (bbox_res["confidence"] >= 0.80 and (proj_res["confidence"] < 0.85 or bbox_res["confidence"] > proj_res["confidence"])):
                return {
                    "orientation": bbox_res["orientation"],
                    "confidence": bbox_res["confidence"],
                    "method": "bbox_override",
                    "projection": proj_res,
                    "bbox": bbox_res,
                }

    return {
        "orientation": proj_res["orientation"] if proj_res["orientation"] != "ambiguous" else "horizontal",
        "confidence": proj_res["confidence"],
        "method": "projection_primary",
        "projection": proj_res,
        "bbox": None,
    }


def collect_book_sample_pages(
    inputs: Union[str, Path, List[Union[str, Path]]],
    sample_count: int = 5,
    recursive: bool = True,
) -> Tuple[List[Path], str]:
    """
    從指定路徑或目錄收集圖片，並嚴格以 Page-xxx.jpg 優先規則進行等距抽樣。
    優先順序：
      1. 第一優先級：檔名符合 Page-*.jpg (或 Page-*.png 等正文標記)
      2. 第二優先級：排除明確標記為 Cover-* 或 Content-* 的圖檔
      3. 第三優先級：所有可用之圖檔
    回傳: (抽樣圖檔清單, 抽樣來源說明)
    """
    raw_inputs = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
    all_images: List[Path] = []
    seen = set()

    exclude_dir_names = {".git", "images", "output", "pipeline_temp", "images_opt_tmp", "__pycache__", "build", "deb_dist"}

    for item in raw_inputs:
        p = Path(item).resolve()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            if str(p) not in seen:
                seen.add(str(p))
                all_images.append(p)
        elif p.is_dir():
            if recursive:
                for root, dirs, files in os.walk(p):
                    dirs[:] = [d for d in dirs if d not in exclude_dir_names and not d.startswith(".")]
                    for f in files:
                        fp = Path(root) / f
                        if fp.suffix.lower() in VALID_IMAGE_EXTS and str(fp) not in seen:
                            seen.add(str(fp))
                            all_images.append(fp)
            else:
                for fp in p.iterdir():
                    if fp.is_file() and fp.suffix.lower() in VALID_IMAGE_EXTS and str(fp) not in seen:
                        seen.add(str(fp))
                        all_images.append(fp)

    if not all_images:
        return [], "未找到任何有效圖檔"

    # [核心規則] 1. 優先挑選 Page-xxx.jpg 正文頁面
    page_imgs = [
        p for p in all_images
        if re.search(r"page[-_]?\d+", p.name, re.IGNORECASE)
    ]
    if len(page_imgs) >= 3:
        target_pool = page_imgs
        pool_desc = f"正文頁面 (Page-*.jpg, 共 {len(page_imgs)} 頁)"
    else:
        # 2. 排除 Cover 與 Content
        non_cover_content = [
            p for p in all_images
            if not (
                p.name.lower().startswith("cover-")
                or p.name.lower().startswith("content-")
                or re.search(r"^(cover|content)[-_]?\d+", p.name, re.IGNORECASE)
            )
        ]
        if len(non_cover_content) >= 1:
            target_pool = non_cover_content
            pool_desc = f"非封面/非目錄頁面 (共 {len(non_cover_content)} 頁)"
        else:
            target_pool = all_images
            pool_desc = f"全部可用圖檔 (共 {len(all_images)} 頁)"

    # 自然排序
    target_pool.sort(key=natural_sort_key)

    # 等距抽樣 (避開極端邊界，取正文 15% ~ 85% 區間)
    n = len(target_pool)
    if n <= sample_count:
        sampled_paths = target_pool
    else:
        indices = np.linspace(int(n * 0.15), int(n * 0.85), sample_count, dtype=int)
        sampled_paths = [target_pool[idx] for idx in sorted(list(set(indices)))]

    return sampled_paths, pool_desc


def detect_book_orientation(
    inputs: Union[str, Path, List[Union[str, Path]]],
    sample_count: int = 5,
    sample_width: int = 800,
    threshold_vertical_ratio: float = 1.35,
    threshold_horizontal_ratio: float = 0.75,
    short_label: bool = True,
) -> Dict[str, Any]:
    """
    書籍層級排版方向自動偵測核心函式：
    抽樣以 Page-xxx.jpg 為主，回傳包含完整抽樣統計與決策結果。
    """
    sampled_paths, pool_desc = collect_book_sample_pages(inputs, sample_count=sample_count)

    if not sampled_paths:
        default_ori = "horizon" if short_label else "horizontal"
        return {
            "orientation": default_ori,
            "confidence": 0.0,
            "votes": {"vertical": 0, "horizontal": 0, "ambiguous": 0},
            "total_sampled": 0,
            "pool_desc": pool_desc,
            "samples": [],
            "error": "未找到可用於抽樣之圖檔",
        }

    votes = {"vertical": 0, "horizontal": 0, "ambiguous": 0}
    sample_details = []

    for sp in sampled_paths:
        res = detect_orientation_projection(
            sp,
            sample_width=sample_width,
            threshold_vertical_ratio=threshold_vertical_ratio,
            threshold_horizontal_ratio=threshold_horizontal_ratio,
        )
        votes[res["orientation"]] += 1
        sample_details.append({
            "path": str(sp),
            "name": sp.name,
            "orientation": res["orientation"],
            "confidence": res["confidence"],
            "ratio": res["ratio"],
            "var_h": res["var_h"],
            "var_v": res["var_v"],
        })

    v_votes = votes["vertical"]
    h_votes = votes["horizontal"]
    decisive_total = v_votes + h_votes

    if decisive_total > 0 and (v_votes / decisive_total) >= 0.5:
        final_orientation = "vertical"
        confidence = round(v_votes / decisive_total, 3)
    else:
        final_orientation = "horizon" if short_label else "horizontal"
        confidence = round((h_votes / max(1, decisive_total)), 3)

    return {
        "orientation": final_orientation,
        "confidence": confidence,
        "votes": votes,
        "total_sampled": len(sampled_paths),
        "pool_desc": pool_desc,
        "samples": sample_details,
    }


def detect_orientation(
    input_target: Union[str, Path, List[Union[str, Path]]],
    sample_count: int = 5,
    short_label: bool = True,
    sample_width: int = 800,
    threshold_vertical_ratio: float = 1.35,
    threshold_horizontal_ratio: float = 0.75,
) -> str:
    """
    主要高階 API：直接回傳 'vertical' 或 'horizon' (若 short_label=False 則為 'horizontal') 字串。
    供 batch_process.py, process_book.py 或外部工具直接呼叫。
    """
    res = detect_book_orientation(
        inputs=input_target,
        sample_count=sample_count,
        sample_width=sample_width,
        threshold_vertical_ratio=threshold_vertical_ratio,
        threshold_horizontal_ratio=threshold_horizontal_ratio,
        short_label=short_label,
    )
    return res["orientation"]


def main():
    parser = argparse.ArgumentParser(
        description="ScanORC: 頁面與書籍排版方向自動偵測工具 (直排 vertical / 橫排 horizon)"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[],
        help="待檢測的書籍目錄、圖片清單或單張圖片 (亦可使用 -i/--input 指定)"
    )
    parser.add_argument(
        "--input", "-i",
        dest="input_target",
        default=None,
        help="明確指定待檢測的書籍圖檔目錄或圖片路徑"
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=5,
        help="書籍抽樣頁數 (預設: 5 頁，優先抽樣 Page-xxx.jpg)"
    )
    parser.add_argument(
        "--sample-width",
        type=int,
        default=800,
        help="投影方差檢測縮放寬度 (預設: 800px，兼具 3~8ms 高速與高準度)"
    )
    parser.add_argument(
        "--threshold-vertical",
        type=float,
        default=1.35,
        help="判定為直排之垂直/水平方差比值閥值 (預設: 1.35)"
    )
    parser.add_argument(
        "--threshold-horizontal",
        type=float,
        default=0.75,
        help="判定為橫排之垂直/水平方差比值閥值 (預設: 0.75)"
    )
    parser.add_argument(
        "--short-label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="回傳橫排標籤時使用 'horizon' 代替 'horizontal' (預設: True，回傳 'vertical' 或 'horizon')"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="精簡模式：標準輸出僅印出結果字串 ('vertical' 或 'horizon'/'horizontal')，便於腳本與子進程捕獲"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式輸出完整抽樣明細、方差比值與信心度"
    )

    args = attach_preference_and_template_support(parser, "orientation_detector")

    # 彙整輸入來源
    input_sources: List[Union[str, Path]] = []
    if args.input_target:
        input_sources.append(args.input_target)
    if args.inputs:
        input_sources.extend(args.inputs)
    if not input_sources:
        # 預設嘗試當前目錄、preprocessed 或 samples
        for cand in ["preprocessed", "samples", "."]:
            if Path(cand).is_dir():
                input_sources.append(cand)
                break

    # 單張特定檔案判定 (若明確給定且只有 1 個檔案)
    single_file_mode = False
    if len(input_sources) == 1:
        single_p = Path(input_sources[0])
        if single_p.is_file() and single_p.suffix.lower() in VALID_IMAGE_EXTS:
            single_file_mode = True

    if single_file_mode:
        res = detect_page_orientation(input_sources[0])
        ori_str = res["orientation"]
        if args.short_label and ori_str == "horizontal":
            ori_str = "horizon"

        if args.quiet:
            print(ori_str)
            sys.exit(0)

        if args.json:
            print(json.dumps({"file": str(input_sources[0]), "result": res, "label": ori_str}, indent=2, ensure_ascii=False))
            sys.exit(0)

        ori_desc = "直排 (豎排 / vertical-rl)" if res["orientation"] == "vertical" else "橫排 (horizontal)"
        print("=" * 60)
        print(f"單頁排版方向檢測: {input_sources[0]}")
        print(f"判定結果: {ori_desc} -> '{ori_str}'")
        print(f"信心度: {res['confidence'] * 100:.1f}%, 投影方差比 (V/H): {res['projection']['ratio']:.3f}")
        print("=" * 60)
        sys.exit(0)

    # 書籍多頁抽樣模式 (以 Page-xxx.jpg 為主)
    book_res = detect_book_orientation(
        inputs=input_sources,
        sample_count=args.samples,
        sample_width=args.sample_width,
        threshold_vertical_ratio=args.threshold_vertical,
        threshold_horizontal_ratio=args.threshold_horizontal,
        short_label=args.short_label,
    )

    res_str = book_res["orientation"]

    if args.quiet:
        print(res_str)
        sys.exit(0)

    if args.json:
        print(json.dumps(book_res, indent=2, ensure_ascii=False))
        sys.exit(0)

    ori_desc = "直排 (豎排 / vertical-rl)" if res_str == "vertical" else "橫排 (horizontal)"
    print("=" * 65)
    print(f"ScanORC 書籍排版方向自動偵測 (抽樣來源: {book_res['pool_desc']})")
    print(f"最終判定結果: {ori_desc} -> '{res_str}'")
    print(f"多數決信心度: {book_res['confidence'] * 100:.1f}% (直排: {book_res['votes']['vertical']} 頁, 橫排: {book_res['votes']['horizontal']} 頁)")
    print("=" * 65)
    print("抽樣檢測明細:")
    for s in book_res["samples"]:
        s_ori = "直排" if s["orientation"] == "vertical" else "橫排"
        print(f"  - {s['name']:<18} -> {s_ori:<4} (比值 V/H: {s['ratio']:<7.3f}, 信心度: {s['confidence']*100:.1f}%)")


if __name__ == "__main__":
    main()
