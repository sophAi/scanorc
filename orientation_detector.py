#!/usr/bin/env python3
"""
ScanORC: 頁面排版方向自動偵測模組 (orientation_detector.py)
結合：
1. 影像水平/垂直投影方差分析 (Projection Profile Variance, PPV)
2. DocLayout-YOLO 文字區塊長寬比幾何統計 (BBox Aspect Ratio Statistics)
3. 書籍層級多數決抽樣共識機制 (Book-level Sampling & Consensus)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Set

import cv2
import numpy as np
from PIL import Image

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


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
    結合投影方差分析與 YOLO BBox 統計的綜合判定。
    """
    proj_res = detect_orientation_projection(image_input)

    if boxes is not None and len(boxes) > 0:
        bbox_res = detect_orientation_bboxes(boxes)
        # 若投影法與 YOLO BBox 一致，大幅提升信心度
        if proj_res["orientation"] == bbox_res["orientation"] and proj_res["orientation"] != "ambiguous":
            return {
                "orientation": proj_res["orientation"],
                "confidence": min(1.0, max(proj_res["confidence"], bbox_res["confidence"]) + 0.1),
                "method": "hybrid_consensus",
                "projection": proj_res,
                "bbox": bbox_res,
            }
        # 若投影法不明確，以 BBox 決策
        if proj_res["orientation"] == "ambiguous" and bbox_res["orientation"] != "ambiguous":
            return {
                "orientation": bbox_res["orientation"],
                "confidence": bbox_res["confidence"],
                "method": "bbox_primary",
                "projection": proj_res,
                "bbox": bbox_res,
            }
        # 若兩者衝突，且一方信心度明顯高於另一方
        if bbox_res["confidence"] > 0.85 and proj_res["confidence"] < 0.7:
            return {
                "orientation": bbox_res["orientation"],
                "confidence": bbox_res["confidence"],
                "method": "bbox_override",
                "projection": proj_res,
                "bbox": bbox_res,
            }

    # 預設以投影方差法為主
    return {
        "orientation": proj_res["orientation"] if proj_res["orientation"] != "ambiguous" else "horizontal",
        "confidence": proj_res["confidence"],
        "method": "projection_primary",
        "projection": proj_res,
        "bbox": None,
    }


def detect_book_orientation(
    image_paths: List[Union[str, Path]],
    sample_count: int = 5,
) -> Dict[str, Any]:
    """
    書籍層級多數決抽樣共識機制：
    1. 優先挑選內文頁 (Page-*.jpg)，避開可能混排之封面 (Cover) 與目錄 (Content)。
    2. 等距抽取 sample_count (預設 5) 頁執行極速投影方差檢測。
    3. 若直排得票率 >= 50% 則判定全書為直排 (vertical)，否則為橫排 (horizontal)。
    """
    valid_paths = [Path(p) for p in image_paths if Path(p).suffix.lower() in VALID_IMAGE_EXTS]
    if not valid_paths:
        return {
            "orientation": "horizontal",
            "confidence": 0.0,
            "votes": {"vertical": 0, "horizontal": 0},
            "samples": [],
            "reason": "未提供有效圖片路徑",
        }

    # 優先挑選內文頁
    page_imgs = [p for p in valid_paths if "page-" in p.name.lower()]
    target_pool = page_imgs if len(page_imgs) >= 3 else valid_paths

    # 排除 Cover 與 Content
    filtered_pool = [
        p for p in target_pool
        if not p.name.lower().startswith("cover-") and not p.name.lower().startswith("content-")
    ]
    if len(filtered_pool) >= 3:
        target_pool = filtered_pool

    # 等距抽樣
    n = len(target_pool)
    if n <= sample_count:
        sampled_paths = target_pool
    else:
        indices = np.linspace(int(n * 0.15), int(n * 0.85), sample_count, dtype=int)
        sampled_paths = [target_pool[idx] for idx in sorted(list(set(indices)))]

    votes = {"vertical": 0, "horizontal": 0, "ambiguous": 0}
    sample_details = []

    for sp in sampled_paths:
        res = detect_orientation_projection(sp)
        votes[res["orientation"]] += 1
        sample_details.append({
            "path": str(sp),
            "name": sp.name,
            "orientation": res["orientation"],
            "confidence": res["confidence"],
            "ratio": res["ratio"],
        })

    v_votes = votes["vertical"]
    h_votes = votes["horizontal"]
    decisive_total = v_votes + h_votes

    if decisive_total > 0 and (v_votes / decisive_total) >= 0.5:
        final_orientation = "vertical"
        confidence = round(v_votes / decisive_total, 3)
    else:
        final_orientation = "horizontal"
        confidence = round((h_votes / max(1, decisive_total)), 3)

    return {
        "orientation": final_orientation,
        "confidence": confidence,
        "votes": votes,
        "total_sampled": len(sampled_paths),
        "samples": sample_details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ScanORC: 頁面與書籍排版方向自動偵測 (直排 / 橫排)"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="單張/多張圖片路徑，或包含圖片之書籍目錄"
    )
    parser.add_argument(
        "--dir", "-d",
        default=None,
        help="待檢測的書籍圖片目錄"
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=5,
        help="書籍抽樣頁數 (預設: 5)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式輸出詳細分析結果"
    )

    args = parser.parse_args()

    input_paths: List[Path] = []
    if args.dir:
        dp = Path(args.dir)
        if dp.is_dir():
            for root, _, files in os.walk(dp):
                for f in files:
                    if Path(f).suffix.lower() in VALID_IMAGE_EXTS:
                        input_paths.append(Path(root) / f)
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            for root, _, files in os.walk(p):
                for f in files:
                    if Path(f).suffix.lower() in VALID_IMAGE_EXTS:
                        input_paths.append(Path(root) / f)
        elif p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            input_paths.append(p)

    if not input_paths:
        parser.print_help()
        print("\n[-] 請指定至少一張圖片或圖片目錄！", file=sys.stderr)
        sys.exit(1)

    import json
    if len(input_paths) == 1:
        # 單張圖片模式
        res = detect_page_orientation(input_paths[0])
        if args.json:
            print(json.dumps({"file": str(input_paths[0]), "result": res}, indent=2, ensure_ascii=False))
        else:
            ori_text = "直排 (豎排 / Vertical-RL)" if res["orientation"] == "vertical" else "橫排 (Horizontal)"
            print(f"檔案: {input_paths[0]}")
            print(f"判定結果: {ori_text} (信心度: {res['confidence'] * 100:.1f}%, 投影方差比 V/H: {res['projection']['ratio']:.3f})")
    else:
        # 多張 / 書籍模式
        book_res = detect_book_orientation(input_paths, sample_count=args.samples)
        if args.json:
            print(json.dumps(book_res, indent=2, ensure_ascii=False))
        else:
            ori_text = "直排 (豎排 / Vertical-RL)" if book_res["orientation"] == "vertical" else "橫排 (Horizontal)"
            print("=" * 60)
            print(f"書籍排版方向檢測結果: {ori_text}")
            print(f"多數決信心度: {book_res['confidence'] * 100:.1f}% (直排: {book_res['votes']['vertical']} 頁, 橫排: {book_res['votes']['horizontal']} 頁)")
            print("=" * 60)
            print("抽樣檢測明細:")
            for s in book_res["samples"]:
                s_ori = "直排" if s["orientation"] == "vertical" else "橫排"
                print(f"  - {s['name']:<18} -> {s_ori:<4} (比值 V/H: {s['ratio']:<7.3f}, 信心度: {s['confidence']*100:.1f}%)")


if __name__ == "__main__":
    main()
