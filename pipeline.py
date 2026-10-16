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
import fnmatch
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import torch
from PIL import Image
try:
    from doclayout_yolo import YOLOv10
except ImportError:
    YOLOv10 = None

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
    from recompress_images import recompress_directory
except ImportError:
    recompress_directory = None

try:
    from orientation_detector import detect_book_orientation, detect_page_orientation, normalize_layout_direction
except ImportError:
    detect_book_orientation = None
    detect_page_orientation = None

    def normalize_layout_direction(val: Optional[str]) -> str:
        if not val:
            return "auto"
        s = str(val).strip().lower()
        if s in ("vertical", "vert", "vertical-rl", "v"):
            return "vertical"
        if s in ("horizontal", "horizon", "horizontal-tb", "h"):
            return "horizontal"
        return "auto"


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
    if re.search(r"(^cover|^封面|^front[-_]?cover|^content)", stem):
        return 0

    # 正文頁碼判斷 (Page-xxx, p-xxx, p0xx, 純數字檔名等)
    if re.search(r"(^page|^p[-_]?\d+|^\d+$|頁|页)", stem):
        return 1

    return 2


def find_default_cover_image(search_dirs: List[Path]) -> Optional[Path]:
    """
    自動搜尋預設封面圖檔 Cover-01.jpg (相容 cover-01.jpg, Cover-01.png, Cover-01.webp 等)
    若無 Cover-01.jpg 則嘗試尋找通用 cover.* 圖檔
    """
    cover01_pattern = re.compile(r"^[Cc]over[-_]?0*1\.(jpg|jpeg|png|webp|tif|tiff)$", re.IGNORECASE)
    fallback_pattern = re.compile(r"^[Cc]over\.(jpg|jpeg|png|webp|tif|tiff)$", re.IGNORECASE)

    # 第一階段：嚴格優先比對 Cover-01.jpg
    for d in search_dirs:
        if not d.exists():
            continue
        if d.is_file():
            if cover01_pattern.match(d.name):
                return d.resolve()
            continue
        # 目錄本層優先
        try:
            for f in sorted(d.iterdir()):
                if f.is_file() and cover01_pattern.match(f.name):
                    return f.resolve()
        except OSError:
            pass
        # 子目錄遞迴
        for root, _, files in os.walk(d):
            for f in sorted(files):
                if cover01_pattern.match(f):
                    return (Path(root) / f).resolve()

    # 第二階段：退而求其次比對 cover.jpg
    for d in search_dirs:
        if not d.exists():
            continue
        if d.is_file():
            if fallback_pattern.match(d.name):
                return d.resolve()
            continue
        try:
            for f in sorted(d.iterdir()):
                if f.is_file() and fallback_pattern.match(f.name):
                    return f.resolve()
        except OSError:
            pass
        for root, _, files in os.walk(d):
            for f in sorted(files):
                if fallback_pattern.match(f):
                    return (Path(root) / f).resolve()

    return None


def derive_title_from_cover(cover_path: Path) -> str:
    """
    以 Cover-01.jpg 所在的目錄名稱作為 title。
    若目錄為 preprocessed、preprocessed_samples 等中間處理資料夾，向上取上一層真實書籍目錄名稱。
    """
    cover_dir = cover_path.resolve().parent
    if cover_dir.name.lower() in ("preprocessed", "preprocessed_samples") and cover_dir.parent.name:
        if cover_dir.parent.name.lower() != "output":
            return cover_dir.parent.name
        elif cover_dir.parent.parent.name:
            return cover_dir.parent.parent.name
    return cover_dir.name or "數位化掃描書本"


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
    cover_first: bool = True,
    exclude_dirs: Optional[List[Path]] = None
) -> List[Path]:
    """
    收集並編排所有輸入圖檔：
    1. 支援單一檔案、多個檔案、目錄路徑或萬用字元
    2. 支援遞迴 (recursive) 讀取所有子目錄
    3. 依據 Cover-xx.jpg 與 Page-xxx.jpg 規則排序 (預設 Cover 在前)
    4. 自動排除輸出目錄與建置暫存目錄
    """
    import glob

    collected: List[Path] = []
    seen = set()

    # 解析所有明確給定的輸入路徑 (不含萬用字元)
    resolved_inputs = [
        Path(inp).resolve()
        for inp in inputs
        if not any(char in inp for char in ["*", "?", "["])
    ]

    norm_exclude: List[Path] = []
    if exclude_dirs:
        for ed in exclude_dirs:
            ed_r = ed.resolve()
            # 若排除目錄剛好是某些輸入目錄的本身或祖先 (例如 --output 是 --samples 的父目錄)，
            # 絕不能直接排除該祖先，否則會把輸入目錄的所有檔案全部略過。
            # 此時僅排除該目錄底下的輸出子目錄 (如 images, images_opt_tmp, pipeline_temp)
            is_ancestor_or_same = any(inp == ed_r or ed_r in inp.parents for inp in resolved_inputs)
            if not is_ancestor_or_same:
                norm_exclude.append(ed_r)
            else:
                for sub in ["images", "images_opt_tmp", "pipeline_temp", "html"]:
                    norm_exclude.append(ed_r / sub)

    def is_excluded(path: Path) -> bool:
        r_p = path.resolve()
        for ed in norm_exclude:
            if r_p == ed or ed in r_p.parents:
                return True
        return False

    for arg in inputs:
        # 萬用字元支援
        if any(char in arg for char in ["*", "?", "["]):
            matched = glob.glob(arg, recursive=recursive)
            for m in matched:
                mp = Path(m).resolve()
                if is_excluded(mp):
                    continue
                if mp.is_file() and mp.suffix.lower() in VALID_IMAGE_EXTS and str(mp) not in seen:
                    seen.add(str(mp))
                    collected.append(mp)
            continue

        p = Path(arg).resolve()
        if is_excluded(p):
            continue
        if p.is_file():
            if p.suffix.lower() in VALID_IMAGE_EXTS and str(p) not in seen:
                seen.add(str(p))
                collected.append(p)
            elif p.suffix.lower() not in VALID_IMAGE_EXTS:
                print(f"[-] 略過非支援之圖片檔案: {p.name}", file=sys.stderr)
        elif p.is_dir():
            if recursive:
                for root, dirs, files in os.walk(p):
                    dirs[:] = [
                        dn for dn in dirs
                        if not dn.startswith(".")
                        and dn not in ("images_opt_tmp", "deb_dist", "build", "dist", "__pycache__")
                        and not is_excluded(Path(root) / dn)
                    ]
                    if is_excluded(Path(root)):
                        continue
                    for f in files:
                        fp = Path(root) / f
                        if is_excluded(fp):
                            continue
                        if fp.suffix.lower() in VALID_IMAGE_EXTS and str(fp) not in seen:
                            seen.add(str(fp))
                            collected.append(fp)
            else:
                for fp in p.iterdir():
                    if is_excluded(fp):
                        continue
                    if fp.is_file() and fp.suffix.lower() in VALID_IMAGE_EXTS and str(fp) not in seen:
                        seen.add(str(fp))
                        collected.append(fp)
        else:
            parent = p.parent
            pattern = p.name
            if parent.exists() and not is_excluded(parent):
                matches = list(parent.glob(pattern))
                for m in matches:
                    if is_excluded(m):
                        continue
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
    以及萬用字元樣式 (wildcards, 例如 Cover-*.jpg, Content-*.jpg, Page-0[1-5]*)
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
                            line_p = Path(line)
                            line_lower = line.lower()
                            targets.add(line_lower)
                            targets.add(line_p.name.lower())
                            targets.add(line_p.stem.lower())
                            if not any(c in line for c in ("*", "?", "[")):
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
        targets.add(p.name.lower())
        targets.add(p.stem.lower())
        if not any(c in item_str for c in ("*", "?", "[")):
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
    比對完整檔名、stem、結尾數字頁碼或萬用字元樣式 (wildcards, 如 Cover-*.jpg, Content-*)
    """
    if not skip_targets:
        return False
    name_lower = img_path.name.lower()
    stem_lower = img_path.stem.lower()

    # 1. 精確比對檔名或 stem
    if name_lower in skip_targets or stem_lower in skip_targets:
        return True

    # 2. 比對結尾數字頁碼
    nums = re.findall(r"\d+", stem_lower)
    if nums and (nums[-1] in skip_targets or str(int(nums[-1])) in skip_targets):
        return True

    # 3. 萬用字元樣式比對 (支援 * 與 ? 等 glob pattern)
    for target in skip_targets:
        if any(c in target for c in ("*", "?", "[")):
            if fnmatch.fnmatch(name_lower, target) or fnmatch.fnmatch(stem_lower, target):
                return True

    return False


def resolve_model_paths(models_dir: Optional[str] = None, yolo_weights: Optional[str] = None, glm_model_dir: Optional[str] = None):
    """解析 AI 模型權重路徑 (優先順序: 明確指定 > --models-dir > SCANORC_MODELS_DIR > 專案內置 models)"""
    script_dir = Path(__file__).resolve().parent
    env_models_dir = os.environ.get("SCANORC_MODELS_DIR")

    if models_dir:
        base_models = Path(models_dir).resolve()
    elif env_models_dir:
        base_models = Path(env_models_dir).resolve()
    elif (script_dir / "models").exists():
        base_models = (script_dir / "models").resolve()
    elif Path("/opt/models").exists():
        base_models = Path("/opt/models").resolve()
    else:
        base_models = (script_dir / "models").resolve()

    if not yolo_weights:
        yolo_pt = base_models / "doclayout_yolo" / "doclayout_yolo_docstructbench_imgsz1024.pt"
        if not yolo_pt.exists():
            pts = list((base_models / "doclayout_yolo").glob("*.pt")) if (base_models / "doclayout_yolo").exists() else []
            if pts:
                yolo_pt = pts[0]
        yolo_weights = str(yolo_pt)

    if not glm_model_dir:
        glm_path = base_models / "glm-ocr"
        glm_model_dir = str(glm_path)

    return str(yolo_weights), str(glm_model_dir)


class DocumentConverter:
    def __init__(
        self,
        models_dir: Optional[str] = None,
        yolo_weights: Optional[str] = None,
        glm_model_dir: Optional[str] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        max_ocr_dim: int = 1024,
        batch_size: int = 2,
        layout_direction: str = "auto",
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 10,
    ):
        self.resolved_yolo, self.resolved_glm = resolve_model_paths(models_dir, yolo_weights, glm_model_dir)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self.max_ocr_dim = max_ocr_dim
        self.batch_size = max(1, batch_size)
        self.layout_direction = layout_direction
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.yolo = None
        self.glm = None
        self.processor = None

    def _ensure_models_loaded(self):
        if self.yolo is not None and self.glm is not None:
            return
        if YOLOv10 is None:
            raise RuntimeError("未安裝 doclayout_yolo 套件，請先執行 'scanorc-setup' 或 'pip install doclayout_yolo'！")
        if GlmOcrForConditionalGeneration is None:
            raise RuntimeError("未安裝支援 GLM-OCR 的 transformers 套件，請先執行 'scanorc-setup'！")
        if self.yolo is None:
            print(f"[*] 載入 DocLayout-YOLO 模型 ({self.resolved_yolo}) ({self.device})...")
            self.yolo = YOLOv10(self.resolved_yolo)
        if self.glm is None:
            print(f"[*] 載入 GLM-OCR 0.9B 模型與處理器 ({self.resolved_glm}) ({self.device}, {self.dtype}, batch_size={self.batch_size})...")
            self.processor = AutoProcessor.from_pretrained(self.resolved_glm, trust_remote_code=True)
            self.glm = GlmOcrForConditionalGeneration.from_pretrained(
                self.resolved_glm,
                dtype=self.dtype,
                device_map=self.device,
                trust_remote_code=True
            )
            self.glm.eval()
            print(f"[+] 所有模型就緒！目前 VRAM 佔用: {round(torch.cuda.memory_allocated() / (1024**2), 1)} MB")

    @staticmethod
    def _has_text_content(im: Image.Image, threshold: int = 210, min_ratio: float = 0.003) -> bool:
        """
        快速檢驗影像是否含有實質文字筆畫特徵 (非純白空頁，深色像素大於指定比例)
        """
        try:
            import numpy as np
            gray = im.convert("L")
            arr = np.array(gray)
            dark_ratio = float(np.mean(arr < threshold))
            return dark_ratio >= min_ratio
        except Exception:
            return True

    def run_layout_analysis(
        self,
        image_path: str,
        conf: float = 0.25,
        layout_direction: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        self._ensure_models_loaded()
        im = Image.open(image_path)
        img_w, img_h = im.size

        def _parse_and_sanitize_boxes(results):
            boxes = []
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
                    x1, y1, x2, y2 = xyxy
                    bw = x2 - x1
                    bh = y2 - y1

                    # 校正誤判為 abandon 的主要文字區塊：
                    # 真正的頁眉通常在極頂部 (y1 <= 10% 頁高 且高度 <= 160)；頁碼在極底端 (y2 >= 90% 頁高 且高度 <= 120)
                    # 若為直排長條文字 (bh > 200) 或遠離天頭/地腳的孤立行，絕非頁眉/頁碼，校正為 title 或 plain text
                    if cls_name == "abandon":
                        is_near_top_edge = (y1 <= int(img_h * 0.10) and bh <= 160)
                        is_near_bottom_edge = (y2 >= int(img_h * 0.90) and bh <= 120)
                        if not is_near_top_edge and not is_near_bottom_edge:
                            if bh > 200 or (bw > 200 and bh > 60):
                                cls_name = "title" if bw <= 100 else "plain text"
                                print(f"  [!] 區塊校正：長條文字 {xyxy} 遠離天頭/地腳 (H={bh})，由 abandon 校正為 {cls_name}")

                    boxes.append({
                        "cls_id": cls_id,
                        "cls_name": cls_name,
                        "conf": conf_val,
                        "bbox": xyxy,
                    })
            return boxes

        # 第一階段：以標準 conf (預設 0.25) 進行版面分析預測
        results = self.yolo.predict(image_path, imgsz=1024, conf=conf, device=self.device)
        boxes_info = _parse_and_sanitize_boxes(results)

        # 檢查是否有任何非 abandon 的有效內容區塊
        has_content = any(b["cls_name"] != "abandon" for b in boxes_info)

        # 機制 1：動態兩階段 Confidence 重試 (若第一階段未抓到任何有效內容區塊且 conf > 0.15)
        if not has_content and conf > 0.15:
            retry_conf = 0.15
            print(f"  [*] 第一階段未偵測到有效內文區塊，自動啟動第二階段寬鬆重試 (conf: {retry_conf:.2f})...")
            retry_results = self.yolo.predict(image_path, imgsz=1024, conf=retry_conf, device=self.device)
            retry_boxes = _parse_and_sanitize_boxes(retry_results)
            if any(b["cls_name"] != "abandon" for b in retry_boxes):
                boxes_info = retry_boxes
                print(f"  [+] 第二階段重試成功偵測到 {len(boxes_info)} 個候選區塊")

        effective_direction = layout_direction or self.layout_direction
        if effective_direction == "auto" and detect_page_orientation is not None:
            try:
                page_res = detect_page_orientation(image_path, boxes=boxes_info)
                effective_direction = page_res["orientation"]
            except Exception:
                effective_direction = "horizontal"

        filtered_boxes = self._filter_contained_boxes(boxes_info)
        sorted_boxes = self._sort_reading_order(filtered_boxes, layout_direction=effective_direction)
        return sorted_boxes

    def _filter_contained_boxes(self, boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not boxes:
            return []

        # 檢查本頁面是否真正含有表格 (table)
        has_actual_table = any(b["cls_name"] == "table" for b in boxes)

        filtered = []
        for i, b1 in enumerate(boxes):
            is_contained = False
            for j, b2 in enumerate(boxes):
                if i != j:
                    if self._is_mostly_inside(b1["bbox"], b2["bbox"]):
                        # 情況 1：同類別包含 -> 捨棄較小的 b1
                        if b1["cls_name"] == b2["cls_name"]:
                            is_contained = True
                            break
                        # 情況 2：無表格時的 table_footnote / table_caption 與 plain text / title 高度重疊
                        if not has_actual_table and b1["cls_name"] in ("table_footnote", "table_caption") and b2["cls_name"] in ("plain text", "title"):
                            is_contained = True
                            break
                        # 情況 3：若重疊嚴重且 b2 具有更具體的正文屬性或更高置信度
                        if b1["cls_name"] not in ("plain text", "title") and b2["cls_name"] in ("plain text", "title"):
                            is_contained = True
                            break
                        elif b1["cls_name"] == "abandon" and b2["cls_name"] != "abandon":
                            is_contained = True
                            break
                        elif b1["conf"] < b2["conf"] and self._is_mostly_inside(b2["bbox"], b1["bbox"]):
                            # 兩者幾近完全一致 (相互包含)，保留置信度高者
                            is_contained = True
                            break
            if not is_contained:
                filtered.append(b1)
        return filtered

    def _is_mostly_inside(self, box1: List[int], box2: List[int], overlap_ratio: float = 0.80) -> bool:
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
            return (inter_area / area1) > overlap_ratio
        return False

    def _sort_reading_order(
        self,
        boxes: List[Dict[str, Any]],
        layout_direction: Optional[str] = None,
        col_bin: int = 40,
        row_bin: int = 30,
    ) -> List[Dict[str, Any]]:
        if not boxes:
            return []
        effective_direction = layout_direction or self.layout_direction
        if effective_direction == "vertical":
            # 直排 (豎排) 閱讀順序：由右至左分欄 (以右邊界 x2 負值排序)，同欄內由上而下 (y1 排序)
            return sorted(boxes, key=lambda b: (-(b["bbox"][2] // col_bin), b["bbox"][1]))
        else:
            # 橫排閱讀順序：由上而下分列 (y1 // row_bin)，同列內由左至右 (x1 排序)
            return sorted(boxes, key=lambda b: (b["bbox"][1] // row_bin, b["bbox"][0]))

    def recognize_crops_batch(self, crop_items: List[Tuple[Image.Image, str]]) -> List[str]:
        """
        將多個裁切區塊以 batch_size 分批送入 GLM-OCR 進行平行推論。
        crop_items: List of (PIL.Image, task_type: 'text' | 'formula' | 'table')
        """
        if not crop_items:
            return []

        self._ensure_models_loaded()

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
            gen_kwargs = {
                "max_new_tokens": max_tokens,
            }
            if self.repetition_penalty is not None and self.repetition_penalty > 1.0:
                gen_kwargs["repetition_penalty"] = self.repetition_penalty
            if self.no_repeat_ngram_size is not None and self.no_repeat_ngram_size > 0:
                gen_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size

            with torch.no_grad():
                outputs = self.glm.generate(**inputs, **gen_kwargs)

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

            # 機制 2：若此頁完全未識別出任何有效內容，但檢測到包含實質文字筆畫特徵，啟動【保底整頁 OCR】
            if not final_blocks:
                if self._has_text_content(orig_im):
                    print(f"  [*] 該頁未分割出有效區塊，但包含文字特徵，啟動【保底整頁 OCR】補救辨識...")
                    try:
                        full_ocr_res = self.recognize_crops_batch([(orig_im, "text")])
                        full_text = full_ocr_res[0].strip() if full_ocr_res else ""
                        # 若辨識出實質文字 (大於等於 5 個字)
                        if full_text and len(full_text) >= 5:
                            print(f"  [+] 【保底整頁 OCR】補救成功！抽取 {len(full_text)} 字正文內容！")
                            # 依據換行格式整理為段落
                            raw_paras = [p.strip() for p in full_text.split("\n\n") if p.strip()]
                            if not raw_paras:
                                raw_paras = [full_text]
                            for para in raw_paras:
                                # 若首行為短標題 (小於等於 25 字且無標點符號結尾)，可標記為 title
                                if para == raw_paras[0] and len(para) <= 25 and not para.endswith(("。", "！", "？", "，", "、", ".", "!", "?")):
                                    final_blocks.append({"type": "title", "content": para})
                                else:
                                    final_blocks.append({"type": "text", "content": para})
                    except Exception as e_full_ocr:
                        print(f"  [-] 保底整頁 OCR 執行異常: {e_full_ocr}", file=sys.stderr)

            # 若經保底整頁 OCR 後依然無有效文字內容（代表為純插圖、無字隔頁或空白頁），才自動轉為原圖嵌入
            if not final_blocks:
                print(f"  [-] 該頁確認無內文區塊，自動將原圖嵌入為整頁插圖")
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
        toc_data: Optional[Dict[str, Any]] = None,
        cover_image: Optional[Union[str, Path]] = None,
        layout_direction: str = "horizontal",
    ) -> str:
        body_content = []

        available_page_files = [p["page_id"] for p in page_results]

        # 插入封面區塊 (Cover)
        cover_filename = Path(cover_image).name if cover_image else None
        if cover_filename:
            body_content.append('<section class="book-cover-page" id="book-cover">')
            body_content.append('  <figure class="book-figure fullpage-figure cover-figure">')
            body_content.append(f'    <img src="images/{cover_filename}" alt="封面: {title}" class="cover-image">')
            body_content.append('  </figure>')
            body_content.append('</section>\n<hr class="page-divider">\n')

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
                start_p_val = entry.get("start_page")
                try:
                    start_p = int(start_p_val) if start_p_val is not None and str(start_p_val).strip().lower() != "none" else None
                except (ValueError, TypeError):
                    start_p = None

                end_p_val = entry.get("end_page")
                try:
                    end_p = int(end_p_val) if end_p_val is not None and str(end_p_val).strip().lower() != "none" else start_p
                except (ValueError, TypeError):
                    end_p = start_p

                target_file = None
                if start_p is not None:
                    # 優先 1: 精確比對頁碼
                    for p_num, p_file in page_numbers:
                        if p_num is not None and p_num == start_p:
                            target_file = p_file
                            break
                    # 優先 2: 落在章節頁碼範圍內的第一個頁面
                    if not target_file and end_p is not None:
                        for p_num, p_file in page_numbers:
                            if p_num is not None and start_p <= p_num <= end_p:
                                target_file = p_file
                                break
                    # 優先 3: 依檔名規則推算 fallback
                    if not target_file:
                        target_file = resolve_page_filename(start_p, available_page_files)

                entry_info = dict(entry)
                entry_info["_chapter_index"] = c_idx
                if target_file:
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
                start_p_val = entry.get("start_page")
                try:
                    start_p = int(start_p_val) if start_p_val is not None and str(start_p_val).strip().lower() != "none" else None
                except (ValueError, TypeError):
                    start_p = None

                full_t = entry.get("full_title", entry.get("title", ""))
                target_file = resolve_page_filename(start_p, available_page_files) if start_p is not None else ""

                body_content.append('    <li class="toc-item">')
                if start_p is not None and target_file:
                    body_content.append(f'      <a href="#chapter-{c_idx}" class="toc-link">')
                    body_content.append(f'        <span class="toc-chapter">{full_t}</span>')
                    body_content.append('        <span class="toc-dots"></span>')
                    body_content.append(f'        <span class="toc-page">第 {start_p} 頁 ({target_file})</span>')
                    body_content.append('      </a>')
                else:
                    body_content.append('      <span class="toc-link noclick">')
                    body_content.append(f'        <span class="toc-chapter">{full_t}</span>')
                    body_content.append('      </span>')
                body_content.append('    </li>')

            body_content.append('  </ul>')
            body_content.append('</nav>\n<hr class="page-divider">\n')

        for p_res in page_results:
            page_id = p_res["page_id"]
            if cover_filename and Path(page_id).name.lower() == cover_filename.lower():
                # 封面已於文檔開頭作為獨立封面展示，略過在內文中重複輸出
                continue
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

        body_html_str = "\n".join(body_content)

        body_class = ' class="layout-vertical"' if layout_direction == "vertical" else ''
        vertical_css = """
    body.layout-vertical {
      writing-mode: vertical-rl;
      -webkit-writing-mode: vertical-rl;
      -epub-writing-mode: vertical-rl;
      text-orientation: upright;
      -webkit-text-orientation: upright;
      max-width: none;
      height: 90vh;
      overflow-x: auto;
      overflow-y: hidden;
      padding: 30px 40px;
    }
    body.layout-vertical .book-page {
      display: inline-block;
      vertical-align: top;
      height: 85vh;
      margin: 0 1.5em;
      writing-mode: vertical-rl;
      -webkit-writing-mode: vertical-rl;
      -epub-writing-mode: vertical-rl;
    }
    body.layout-vertical p {
      margin: 0 0.8em;
      text-indent: 2em;
      line-height: 2.0;
    }
    body.layout-vertical h1,
    body.layout-vertical h2,
    body.layout-vertical h3 {
      margin: 0 1.2em;
      line-height: 1.4;
    }
    body.layout-vertical .book-cover-page {
      display: inline-block;
      vertical-align: top;
      height: 85vh;
      margin: 0 2em;
    }
    body.layout-vertical .book-toc {
      display: inline-block;
      vertical-align: top;
      height: 85vh;
      margin: 0 2em;
      overflow-y: auto;
    }
""" if layout_direction == "vertical" else ""

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
    .book-cover-page {{
      margin: 0 auto 3em auto;
      text-align: center;
      page-break-after: always;
      break-after: page;
    }}
    .cover-figure {{
      margin: 0 auto;
      max-width: 100%;
    }}
    .cover-image {{
      max-height: 85vh;
      max-width: 100%;
      height: auto;
      box-shadow: 0 8px 24px rgba(0,0,0,0.15);
      border-radius: 4px;
      display: inline-block;
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
    {vertical_css}
  </style>
</head>
<body{body_class}>
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
    layout_direction: str = "horizontal",
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
    if layout_direction == "vertical":
        cmd.extend([
            "--metadata", "page-progression-direction=rtl",
            "--metadata", "dir=rtl",
        ])
    if cover_image and os.path.exists(cover_image):
        cmd.append(f"--epub-cover-image={os.path.abspath(cover_image)}")

    print(f"[*] 執行 Pandoc 編譯 MathML EPUB (工作目錄: {html_dir}):\n    {' '.join(cmd)}")
    subprocess.run(cmd, cwd=html_dir, check=True)
    print(f"[+] EPUB 生成成功 (含原生 MathML 渲染與章節 Outline): {abs_epub_path}")


def _worker_process_page(args_tuple):
    """多進程 Worker 任務函數"""
    task_idx, img_path, output_dir, batch_size, skip_ocr, models_dir, yolo_weights, glm_model_dir, layout_direction, repetition_penalty, no_repeat_ngram_size = args_tuple
    # 每個子進程獨立初始化 Converter (套用傳入之模型目錄與版面排版方向)
    converter = DocumentConverter(
        models_dir=models_dir,
        yolo_weights=yolo_weights,
        glm_model_dir=glm_model_dir,
        batch_size=batch_size,
        layout_direction=layout_direction,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
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
    parser.add_argument("--title", default="數位化掃描書本", help="書名 / 文檔標題 (預設自動以 Cover-01.jpg 所在目錄名稱命名)")
    parser.add_argument("--toc", default=None, help="目錄結構 JSON 檔案路徑 (例如 toc.json，將自動生成目錄索引並連結至各頁檔名)")
    parser.add_argument("--epub", action="store_true", help="是否自動生成 EPUB")
    parser.add_argument(
        "--cover", "--cover-image",
        default=None,
        dest="cover",
        help="指定 EPUB/HTML 封面圖片檔案路徑 (若未指定將預設自動在待處理圖片中尋找 Cover-01.jpg)"
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
    parser.add_argument(
        "--image-unsharp",
        default=None,
        help="images 圖檔銳化參數 (ImageMagick unsharp 語法例如 1x0.55+1.5+0.002，預設: None 關閉)"
    )
    parser.add_argument(
        "--auto-grayscale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="重新壓縮圖檔時自動判斷是否為灰階色調，若無彩色資訊則改以灰階模式壓縮儲存以節省空間 (預設: True，使用 --no-auto-grayscale 保持原樣)"
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
        help="不作 OCR 的檔案清單 (支援萬用字元如 Cover-*.jpg、Content-*、檔名、頁碼或 txt 清單，直接以原始圖檔嵌入 HTML/EPUB)"
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="AI 模型權重母目錄 (若指定則自動尋找 doclayout_yolo/ 與 glm-ocr/；亦可透過 SCANORC_MODELS_DIR 設定)"
    )
    parser.add_argument(
        "--yolo-weights",
        default=None,
        help="自訂 DocLayout-YOLO 權重檔案路徑 (選填，若未指定則自動從 --models-dir 解析)"
    )
    parser.add_argument(
        "--glm-model-dir",
        default=None,
        help="自訂 GLM-OCR 模型權重目錄 (選填，若未指定則自動從 --models-dir 解析)"
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
        except Exception as e:
            print(f"[-] 讀取目錄索引檔案失敗: {e}", file=sys.stderr)

    # 封面圖檔自動解析與書名 (title) 自動推導：
    # 預設以 Cover-01.jpg 作為 epub 與 html 的封面，Cover-01.jpg 所在的目錄名稱作為 title
    cover_search_dirs: List[Path] = []
    for s_arg in sample_inputs:
        sp = Path(s_arg).resolve()
        if sp.is_file():
            cover_search_dirs.append(sp.parent)
        elif sp.is_dir():
            cover_search_dirs.append(sp)
            if sp.parent != sp:
                cover_search_dirs.append(sp.parent)
    cover_search_dirs.append(Path.cwd())
    if Path(args.output).exists():
        cover_search_dirs.append(Path(args.output).resolve())

    actual_cover: Optional[Path] = None
    if args.cover:
        cov_p = Path(args.cover).resolve()
        if cov_p.exists():
            actual_cover = cov_p
        else:
            print(f"[-] 警告：指定的封面圖檔不存在: {args.cover}", file=sys.stderr)

    if not actual_cover:
        actual_cover = find_default_cover_image(cover_search_dirs)

    # 若未指定自訂書名（保持預設或空白），優先以 Cover-01.jpg 所在目錄名稱作為 title
    if args.title in ("數位化掃描書本", "", None):
        if actual_cover:
            detected_title = derive_title_from_cover(actual_cover)
            if detected_title:
                args.title = detected_title
                print(f"[*] 預設以 Cover-01.jpg 所在目錄自動推導書名: '{args.title}'")
        elif toc_data and toc_data.get("book_title"):
            args.title = toc_data["book_title"]
            print(f"[*] 自動套用 toc.json 中的書名: '{args.title}'")

    # 確保封面圖片複製至 output/images/ 目錄，供 HTML 與 EPUB 引用
    if actual_cover and actual_cover.exists():
        images_dir = Path(args.output) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        dest_cover = images_dir / actual_cover.name
        if not dest_cover.exists() or dest_cover.resolve() != actual_cover.resolve():
            shutil.copy2(actual_cover, dest_cover)
            print(f"[*] 複製封面圖檔至圖片目錄: {dest_cover}")

    out_path = Path(args.output).resolve()
    exclude_targets = [
        out_path / "images",
        out_path / "images_opt_tmp",
        out_path / "pipeline_temp",
    ]
    resolved_samples = [
        Path(s).resolve()
        for s in sample_inputs
        if not any(c in s for c in ["*", "?", "["])
    ]
    if not any(sp == out_path or out_path in sp.parents for sp in resolved_samples):
        exclude_targets.append(out_path)

    image_files = collect_book_images(
        inputs=sample_inputs,
        recursive=args.recursive,
        cover_first=args.cover_first,
        exclude_dirs=exclude_targets
    )

    if not image_files:
        print(f"[-] 未找到待處理之圖片檔案！(搜尋路徑: {sample_inputs})", file=sys.stderr)
        sys.exit(1)

    skip_targets = parse_skip_ocr_targets(args.skip_ocr)
    if actual_cover:
        skip_targets.add(actual_cover.name.lower())
        skip_targets.add(actual_cover.stem.lower())
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

    # 自動辨識書籍排版方向 (直排 / 橫排，優先繼承 toc.json，次選抽樣 Page-*.jpg)
    if args.layout_direction == "auto":
        toc_dir = normalize_layout_direction(toc_data.get("layout_direction")) if toc_data else "auto"
        if toc_dir != "auto":
            resolved_layout_direction = toc_dir
            ori_text = "直排 (豎排 / vertical-rl)" if resolved_layout_direction == "vertical" else "橫排 (horizontal)"
            print(f"[*] 從目錄索引 (toc.json) 自動繼承書籍排版方向: {ori_text}")
        elif detect_book_orientation is not None:
            book_ori_res = detect_book_orientation(image_files, sample_count=5)
            raw_direction = book_ori_res["orientation"]
            resolved_layout_direction = normalize_layout_direction(raw_direction)
            if resolved_layout_direction == "auto":
                resolved_layout_direction = "horizontal"
            v_votes = book_ori_res["votes"]["vertical"]
            h_votes = book_ori_res["votes"]["horizontal"]
            conf_pct = book_ori_res["confidence"] * 100
            ori_text = "直排 (豎排 / vertical-rl)" if resolved_layout_direction == "vertical" else "橫排 (horizontal)"
            pool_desc = book_ori_res.get("pool_desc", "正文頁面")
            print(f"[*] 自動辨識書籍版面排版方向 (抽樣來源: {pool_desc}): {ori_text} [多數決共識度: {conf_pct:.1f}%, 直排: {v_votes} 頁, 橫排: {h_votes} 頁]")
        else:
            resolved_layout_direction = "horizontal"
    else:
        resolved_layout_direction = normalize_layout_direction(args.layout_direction)
        if resolved_layout_direction == "auto":
            resolved_layout_direction = "horizontal"
        ori_text = "直排 (豎排 / vertical-rl)" if resolved_layout_direction == "vertical" else "橫排 (horizontal)"
        print(f"[*] 設定書籍版面排版方向: {ori_text}")

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
            (
                idx,
                str(img_path),
                args.output,
                args.batch_size,
                should_skip_ocr(img_path, skip_targets),
                args.models_dir,
                args.yolo_weights,
                args.glm_model_dir,
                resolved_layout_direction,
                args.repetition_penalty,
                args.no_repeat_ngram_size,
            )
            for idx, img_path in enumerate(image_files)
        ]
        with mp.Pool(processes=args.workers) as pool:
            all_page_results = pool.map(_worker_process_page, tasks)
        # 確保結果嚴格依照任務與編排順序排序
        all_page_results.sort(key=lambda x: x["_task_idx"])
    else:
        converter = DocumentConverter(
            models_dir=args.models_dir,
            yolo_weights=args.yolo_weights,
            glm_model_dir=args.glm_model_dir,
            batch_size=args.batch_size,
            layout_direction=resolved_layout_direction,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
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
    full_html = temp_conv.render_html(
        all_page_results,
        title=args.title,
        toc_data=toc_data,
        cover_image=actual_cover,
        layout_direction=resolved_layout_direction,
    )
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
                    unsharp=args.image_unsharp,
                    auto_grayscale=args.auto_grayscale,
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
        if actual_cover and actual_cover.exists():
            print(f"[*] 套用 EPUB 封面圖檔: {actual_cover}")

        epub_out_path = os.path.join(args.output, "book.epub")
        build_epub_with_pandoc(
            html_out_path,
            epub_out_path,
            title=args.title,
            cover_image=str(actual_cover) if actual_cover else None,
            split_level=1,
            layout_direction=resolved_layout_direction,
        )

    print("\n[🎉] 全流程轉換成功完成！")


if __name__ == "__main__":
    main()
