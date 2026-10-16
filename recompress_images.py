#!/usr/bin/env python3
"""
ScanORC: 批次圖片重新壓縮與縮放工具 (recompress_images.py / images_compressed.py)
將指定目錄下的所有圖檔（預設同時掃描 JPG 與 PNG，包含所有層級的子目錄）
以指定的壓縮率與縮放比例重新壓縮，完整維持子目錄結構與檔案相對分布，並輸出至指定的新目錄。

特色：
  1. 完整鏡像：遞迴遍歷所有子目錄，以 1:1 相同階層鏡像重建目錄結構與檔案相對分布。
  2. 同時支援 JPG 與 PNG：預設同時掃描 .jpg, .jpeg, .png 圖檔，依據個別格式自適應最佳壓縮策略。
  3. 彈性品質與縮放：支援自訂品質 (-q / --quality) 與尺寸縮放比例 (-s / --scale, 如 0.5 或 50%)。
     - JPG：套用 lossy DCT 壓縮率、可選色彩抽樣 (4:4:4 / 4:2:0) 與 Huffman 表最佳化。
     - PNG：套用高品質 Lanczos 縮放；品質 < 100 時自動套用調色盤量化降階以劇烈縮減檔案體積 (可選 --png-lossless 保持無損全彩)。
  4. 多組合批次測試：若指定多組品質或縮放值，自動為各組合生成對應子目錄，方便畫質與體積橫向評估。
  5. 出版級重採樣：縮放採用高品質 Lanczos 重採樣演算法，保持細文字與線條邊緣清晰度。
  6. 多核並行加速：內建 ProcessPoolExecutor 多進程並行處理，數百頁圖檔數秒內高速完成。
  7. 安全與元資料：保留 DPI、ICC 色彩設定檔與 EXIF；採用原子性寫入避免中斷產生損壞檔案。
"""

import os
import re
import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image, ImageFilter, ImageChops
try:
    from config_utils import attach_preference_and_template_support
except ImportError:
    attach_preference_and_template_support = None

VALID_EXTS = {".jpg", ".jpeg", ".png"}


def apply_unsharp_filter(im: Image.Image, unsharp: Optional[str]) -> Image.Image:
    """
    對 PIL 影像套用 Unsharp Mask 銳化濾鏡。
    支援 ImageMagick 語法如 '1x0.55+1.5+0.002' (radiusxsigma+amount+threshold)。
    """
    if not unsharp:
        return im
    unsharp_str = str(unsharp).strip()
    if unsharp_str.lower() in ("none", "off", "0", "false"):
        return im

    r_val, p_val, th_val = 1.0, 150, 0
    m = re.match(r"^(\d*(?:\.\d+)?)x(\d*(?:\.\d+)?)\+([0-9.]+)\+([0-9.]+)$", unsharp_str)
    if m:
        rad_s, sig_s, amt_s, th_s = m.groups()
        rad = float(rad_s) if rad_s else 1.0
        sig = float(sig_s) if sig_s else 0.55
        amt = float(amt_s) if amt_s else 1.5
        th = float(th_s) if th_s else 0.002
        r_val = rad if rad > 0 else sig
        p_val = int(round(amt * 100))
        th_val = int(round(th * 255))
    else:
        try:
            val = float(unsharp_str)
            r_val, p_val, th_val = val, 150, 0
        except ValueError:
            r_val, p_val, th_val = 1.0, 150, 0

    return im.filter(ImageFilter.UnsharpMask(radius=r_val, percent=p_val, threshold=th_val))


def is_grayscale_image(
    img_or_path: Union[Image.Image, Path, str],
    is_cover: Optional[bool] = None,
    chroma_threshold: int = 12,
    high_chroma_threshold: int = 25,
    max_chroma_pixels: int = 40,
    max_high_chroma_pixels: int = 15,
) -> bool:
    """
    智慧判斷圖檔是否實質為灰階色調：
    1. 快速優先判斷 (Fast-path)：
       若原始圖檔原生色彩空間已經是灰階 (單通道 'L', '1', 'LA')，
       無需進行任何像素解碼或通道差值運算，直接判定為 True 節省時間。
    2. 封面保護機制 (is_cover=True)：
       只要存在任何非零色差 (max_diff > 0)，一律視為彩色圖，絕不降階為灰階；
       唯有 max_diff == 0 (100% 絕對純黑白) 時才視為灰階。
    3. 100% 絕對純灰階：
       若 max_diff == 0，RGB 各通道完全一致，判定為 True。
    4. 彩色小圖示與高彩度細節防護：
       - 若微弱色差極低 (max_diff <= 5)，視為感光元件量化雜訊，判定為 True。
       - 統計高彩度像素數量 (diff >= high_chroma_threshold，預設 25)：
         若數量超過 max_high_chroma_pixels (預設 15 像素)，表示包含細小彩色標記、印章或雙色標題，判定為 False (保留彩色)。
       - 統計中彩度像素數量 (diff >= chroma_threshold，預設 12)：
         若數量超過 max_chroma_pixels (預設 40 像素)，表示有明顯色調或彩色區塊，判定為 False (保留彩色)。
       - 若上述均未觸發，表示僅有零星雜訊或紙張微弱底色，判定為 True。
    """
    if isinstance(img_or_path, (str, Path)):
        p = Path(img_or_path)
        try:
            with Image.open(p) as im_file:
                # 優先判斷：如果原始圖檔已經是 Gray Color space (L, 1, LA)，直接回傳 True，避免解碼與像素運算
                if im_file.mode in ("L", "1", "LA"):
                    return True
                if is_cover is None:
                    name_lower = p.name.lower()
                    stem_lower = p.stem.lower()
                    is_cover = bool(
                        name_lower.startswith("cover")
                        or "cover" in stem_lower
                        or "backcover" in stem_lower
                    )
                return is_grayscale_image(
                    im_file,
                    is_cover=is_cover,
                    chroma_threshold=chroma_threshold,
                    high_chroma_threshold=high_chroma_threshold,
                    max_chroma_pixels=max_chroma_pixels,
                    max_high_chroma_pixels=max_high_chroma_pixels,
                )
        except Exception:
            return False

    im = img_or_path
    if is_cover is None:
        is_cover = False

    # 優先判斷：若影像本身已是灰階色彩空間 (L, 1, LA)，直接回傳 True
    if im.mode in ("L", "1", "LA"):
        return True
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")

    r, g, b = im.split()[:3]
    d_rg = ImageChops.difference(r, g)
    d_gb = ImageChops.difference(g, b)
    d_rb = ImageChops.difference(r, b)

    ext_rg = d_rg.getextrema()
    ext_gb = d_gb.getextrema()
    ext_rb = d_rb.getextrema()
    max_diff = max(ext_rg[1], ext_gb[1], ext_rb[1])

    # 1. 100% 絕對純灰階
    if max_diff == 0:
        return True

    # 2. 封面保護機制：只要存在任何非零色差，一律保留彩色
    if is_cover:
        return False

    # 3. 極微弱感光元件雜訊容許 (全圖最大通道差 <= 5)
    if max_diff <= 5:
        return True

    # 4. 統計色差直方圖，防護彩色小圖示或高彩度細節
    h_rg = d_rg.histogram()
    h_gb = d_gb.histogram()
    h_rb = d_rb.histogram()

    # 高彩度像素 (diff >= high_chroma_threshold)
    c_high = max(
        sum(h_rg[high_chroma_threshold:]),
        sum(h_gb[high_chroma_threshold:]),
        sum(h_rb[high_chroma_threshold:])
    )
    if c_high > max_high_chroma_pixels:
        return False

    # 中彩度像素 (diff >= chroma_threshold)
    c_mid = max(
        sum(h_rg[chroma_threshold:]),
        sum(h_gb[chroma_threshold:]),
        sum(h_rb[chroma_threshold:])
    )
    if c_mid > max_chroma_pixels:
        return False

    return True


def parse_scale_value(val_str: Union[str, float]) -> float:
    """解析縮放字串 (支援 0.75 或 75% 格式)"""
    if isinstance(val_str, (int, float)):
        return float(val_str)
    s = str(val_str).strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    return float(s)


def format_bytes(num_bytes: int) -> str:
    """將位元組轉換為人類可讀單位 (KB, MB, GB)"""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def _worker_task(args_tuple) -> Dict:
    """
    單一圖檔壓縮 Worker 函數
    """
    (
        src_path_str,
        dst_path_str,
        quality,
        scale,
        max_dim,
        optimize,
        progressive,
        subsampling,
        preserve_metadata,
        png_lossless,
        unsharp,
        auto_grayscale,
    ) = args_tuple

    src_p = Path(src_path_str)
    dst_p = Path(dst_path_str)
    is_png = src_p.suffix.lower() == ".png"
    has_unsharp = bool(unsharp and str(unsharp).strip().lower() not in ("none", "off", "0", "false"))

    try:
        orig_size = src_p.stat().st_size

        with Image.open(src_p) as im:
            orig_w, orig_h = im.size

            # 計算目標尺寸
            new_w, new_h = orig_w, orig_h
            if scale != 1.0:
                new_w = max(1, int(round(orig_w * scale)))
                new_h = max(1, int(round(orig_h * scale)))

            if max_dim and max(new_w, new_h) > max_dim:
                ratio = max_dim / max(new_w, new_h)
                new_w = max(1, int(round(new_w * ratio)))
                new_h = max(1, int(round(new_h * ratio)))

            dst_p.parent.mkdir(parents=True, exist_ok=True)

            if is_png:
                # ---------------- PNG 處理流程 ----------------
                # 色彩模式處理 (保留透明通道 RGBA，調色盤 P 轉為 RGBA 或 RGB)
                if im.mode == "P":
                    if "transparency" in im.info:
                        im = im.convert("RGBA")
                    else:
                        im = im.convert("RGB")
                elif im.mode not in ("RGBA", "RGB", "L", "LA"):
                    im = im.convert("RGBA" if "A" in im.mode else "RGB")

                # 若尺寸改變，執行高品質 Lanczos 重採樣
                if (new_w, new_h) != (orig_w, orig_h):
                    im = im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

                # 銳化濾鏡處理 (在縮放後、調色盤量化前套用)
                if has_unsharp:
                    im = apply_unsharp_filter(im, unsharp)

                # 品質 -q 作用於 PNG：若非 lossless 且品質小於 100，套用調色盤量化降階 (大幅縮減檔案體積)
                if not png_lossless and quality < 100:
                    if im.mode in ("RGB", "RGBA"):
                        colors = max(16, min(256, int(round(16 + (256 - 16) * (quality / 100.0)))))
                        quant_method = Image.Quantize.FASTOCTREE if im.mode == "RGBA" else Image.Quantize.MEDIANCUT
                        im = im.quantize(colors=colors, method=quant_method)

                save_kwargs = {
                    "format": "PNG",
                    "optimize": optimize,
                    "compress_level": 9,
                }

                if preserve_metadata:
                    if "dpi" in im.info:
                        orig_dpi = im.info["dpi"]
                        if scale != 1.0:
                            save_kwargs["dpi"] = (
                                max(72, int(round(orig_dpi[0] * scale))),
                                max(72, int(round(orig_dpi[1] * scale))),
                            )
                        else:
                            save_kwargs["dpi"] = orig_dpi
                    if "icc_profile" in im.info:
                        save_kwargs["icc_profile"] = im.info["icc_profile"]

            else:
                # ---------------- JPG / JPEG 處理流程 ----------------
                converted_to_gray = False
                # 優先判斷：如果原始圖檔已經是 Gray Color space (如 L, 1, LA)，
                # 則完全不需要額外的色差或像素偵測，直接保持或轉換為灰階 (L 模式) 儲存，大幅節省時間
                if im.mode in ("L", "1", "LA"):
                    if im.mode != "L":
                        im = im.convert("L")
                    converted_to_gray = True
                else:
                    # 原圖為彩色模式 (RGB, RGBA, CMYK, P 等)
                    if auto_grayscale:
                        stem_lower = src_p.stem.lower()
                        name_lower = src_p.name.lower()
                        is_cover = bool(
                            name_lower.startswith("cover")
                            or "cover" in stem_lower
                            or "backcover" in stem_lower
                        )
                        if is_grayscale_image(im, is_cover=is_cover):
                            im = im.convert("L")
                            converted_to_gray = True
                        elif im.mode not in ("RGB", "L"):
                            im = im.convert("RGB")
                    elif im.mode not in ("RGB", "L"):
                        im = im.convert("RGB")

                # 若尺寸改變，執行高品質 Lanczos 重採樣
                if (new_w, new_h) != (orig_w, orig_h):
                    im = im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

                # 銳化濾鏡處理 (在縮放後套用)
                if has_unsharp:
                    im = apply_unsharp_filter(im, unsharp)

                save_kwargs = {
                    "format": "JPEG",
                    "quality": quality,
                    "optimize": optimize,
                    "progressive": progressive,
                }

                # 採樣率設定 (僅色彩模式為 RGB 等彩色模式時適用，灰階 L 模式無色度抽樣)
                if im.mode != "L":
                    if subsampling == "444":
                        save_kwargs["subsampling"] = 0
                    elif subsampling == "422":
                        save_kwargs["subsampling"] = 1
                    elif subsampling == "420":
                        save_kwargs["subsampling"] = 2
                    elif subsampling == "keep":
                        orig_sub = im.info.get("subsampling")
                        if orig_sub is not None:
                            save_kwargs["subsampling"] = orig_sub

                # 元資料保留
                if preserve_metadata:
                    if "dpi" in im.info:
                        orig_dpi = im.info["dpi"]
                        if scale != 1.0:
                            save_kwargs["dpi"] = (
                                max(72, int(round(orig_dpi[0] * scale))),
                                max(72, int(round(orig_dpi[1] * scale))),
                            )
                        else:
                            save_kwargs["dpi"] = orig_dpi
                    # 若為 RGB 轉為 L，切勿嵌入原 RGB ICC profile
                    if "icc_profile" in im.info and not converted_to_gray:
                        save_kwargs["icc_profile"] = im.info["icc_profile"]
                    if "exif" in im.info:
                        save_kwargs["exif"] = im.info["exif"]

            # 原子性寫入：先寫入臨時檔再覆蓋
            temp_file = dst_p.with_name(f".{dst_p.name}.tmp_{os.getpid()}_{time.time_ns()}")
            try:
                im.save(temp_file, **save_kwargs)
                temp_file.replace(dst_p)
            finally:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass

        new_size = dst_p.stat().st_size
        return {
            "success": True,
            "src": str(src_p),
            "dst": str(dst_p),
            "orig_size": orig_size,
            "new_size": new_size,
            "orig_dim": (orig_w, orig_h),
            "new_dim": (new_w, new_h),
            "is_png": is_png,
            "is_gray": (im.mode == "L"),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "src": str(src_p),
            "dst": str(dst_p),
            "orig_size": 0,
            "new_size": 0,
            "orig_dim": (0, 0),
            "new_dim": (0, 0),
            "is_png": is_png,
            "error": str(e),
        }


def collect_image_files(input_dir: Path) -> List[Tuple[Path, Path]]:
    """
    遞迴收集指定目錄下的所有 JPG 與 PNG 檔案，並計算相對路徑
    :return: list of (abs_path, rel_path)
    """
    items = []
    for root, _, files in os.walk(input_dir):
        root_path = Path(root)
        for f in files:
            p = root_path / f
            if p.suffix.lower() in VALID_EXTS:
                rel_p = p.relative_to(input_dir)
                items.append((p.resolve(), rel_p))

    # 自然排序
    def natural_sort_key(item):
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(item[1]))]

    items.sort(key=natural_sort_key)
    return items


def recompress_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    qualities: List[int] = [50],
    scales: List[float] = [1.0],
    max_dim: Optional[int] = None,
    workers: Optional[int] = None,
    optimize: bool = True,
    progressive: bool = False,
    subsampling: str = "keep",
    preserve_metadata: bool = True,
    png_lossless: bool = False,
    unsharp: Optional[str] = "1x0.55+1.5+0.002",
    auto_grayscale: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    遞迴重新壓縮指定目錄及其子目錄下的所有 JPG 與 PNG 檔案。
    """
    in_dir = Path(input_dir).resolve()
    out_dir = Path(output_dir).resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"輸入目錄不存在: {in_dir}")

    if in_dir == out_dir:
        raise ValueError("安全防護：輸出目錄不能與輸入目錄完全相同！請指定一個獨立的新目錄。")

    # 1. 掃描圖檔清單 (JPG 與 PNG)
    all_files = collect_image_files(in_dir)
    total_files = len(all_files)

    if total_files == 0:
        if verbose:
            print(f"[-] 在目錄 '{in_dir}' 下未找到任何 JPG 或 PNG 檔案！")
        return {"total_files": 0, "presets": []}

    jpg_cnt = sum(1 for _, p in all_files if p.suffix.lower() in (".jpg", ".jpeg"))
    png_cnt = sum(1 for _, p in all_files if p.suffix.lower() == ".png")

    workers = workers or min(os.cpu_count() or 4, 16)
    is_multi_combo = (len(qualities) > 1) or (len(scales) > 1)

    if verbose:
        print("======================================================================")
        print(" ScanORC: 批次圖片重新壓縮與縮放工具 (recompress_images.py)")
        print("======================================================================")
        print(f"[*] 輸入目錄:    {in_dir}")
        print(f"[*] 輸出目錄:    {out_dir}")
        print(f"[*] 找到圖檔:    {total_files} 個圖檔 (含 {jpg_cnt} 個 JPG, {png_cnt} 個 PNG，涵蓋所有子目錄)")
        print(f"[*] 壓縮品質:    {qualities}")
        print(f"[*] 縮放比例:    {[f'{int(s*100)}%' if s<=10 else f'{s}' for s in scales]}")
        if unsharp and str(unsharp).strip().lower() not in ("none", "off", "0", "false"):
            print(f"[*] 影像銳化:    啟用 ({unsharp})")
        else:
            print(f"[*] 影像銳化:    關閉")
        if auto_grayscale:
            print(f"[*] 自動灰階:    啟用 (自動判斷無色彩圖檔改以灰階壓縮，保護封面與彩色細節)")
        else:
            print(f"[*] 自動灰階:    關閉 (保持原始色彩模式)")
        if max_dim:
            print(f"[*] 最大長寬:    {max_dim} px")
        print(f"[*] 並行 Worker: {workers}")
        print(f"[*] 演算法最佳化: Lanczos 重採樣 | Huffman 最佳化={optimize}")
        if png_lossless:
            print(f"[*] PNG 模式:    無損全彩 (不執行調色盤降階)")
        else:
            print(f"[*] PNG 模式:    自適應調色盤量化壓縮 (隨 -q 品質縮減體積)")
        if dry_run:
            print("[*] 執行模式:    ⚠️ 模擬運行 (Dry Run - 不寫入檔案)")
        print("======================================================================\n")

    overall_results = []
    combo_idx = 0
    total_combos = len(qualities) * len(scales)

    for q in qualities:
        for s in scales:
            combo_idx += 1
            scale_pct = int(round(s * 100)) if s <= 10 else int(round(s))
            combo_name = f"q{q}_s{scale_pct}"

            # 判定該組設定之輸出根目錄
            if is_multi_combo:
                preset_out_dir = out_dir / combo_name
            else:
                preset_out_dir = out_dir

            if verbose:
                print(f"--- [組合 {combo_idx}/{total_combos}] 品質 Q={q} | 縮放 Scale={scale_pct}% ---")
                print(f"    目標子目錄: {preset_out_dir}")

            tasks = []
            for src_p, rel_p in all_files:
                dst_p = preset_out_dir / rel_p
                tasks.append((
                    str(src_p),
                    str(dst_p),
                    q,
                    s,
                    max_dim,
                    optimize,
                    progressive,
                    subsampling,
                    preserve_metadata,
                    png_lossless,
                    unsharp,
                    auto_grayscale,
                ))

            if dry_run:
                print(f"    [Dry Run] 將處理 {len(tasks)} 個檔案，範例路徑:")
                for _, rel_p in all_files[:3]:
                    print(f"      - {in_dir.name}/{rel_p} -> {preset_out_dir.name}/{rel_p}")
                if len(all_files) > 3:
                    print(f"      ... 其餘 {len(all_files) - 3} 個檔案保持相同相對目錄結構")
                print()
                continue

            # 多進程執行任務
            start_time = time.time()
            combo_orig_size = 0
            combo_new_size = 0
            success_cnt = 0
            error_cnt = 0
            errors = []

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_worker_task, t) for t in tasks]
                completed_cnt = 0

                for fut in as_completed(futures):
                    res = fut.result()
                    completed_cnt += 1

                    if res["success"]:
                        success_cnt += 1
                        combo_orig_size += res["orig_size"]
                        combo_new_size += res["new_size"]

                        # 進度即時顯示 (每 10% 或單一少量時印出)
                        if verbose and (total_files <= 20 or completed_cnt % max(1, total_files // 10) == 0 or completed_cnt == total_files):
                            rel_show = Path(res["dst"]).relative_to(preset_out_dir)
                            dim_show = f"[{res['orig_dim'][0]}x{res['orig_dim'][1]} -> {res['new_dim'][0]}x{res['new_dim'][1]}]"
                            ratio = (res["new_size"] - res["orig_size"]) / max(1, res["orig_size"]) * 100
                            fmt_tag = "PNG" if res.get("is_png") else ("JPG/Gray" if res.get("is_gray") else "JPG/Color")
                            print(
                                f"    [{completed_cnt}/{total_files}] [{fmt_tag}] {rel_show} "
                                f"({format_bytes(res['orig_size'])} -> {format_bytes(res['new_size'])}, {ratio:+.1f}%) {dim_show}"
                            )
                    else:
                        error_cnt += 1
                        errors.append((res["src"], res["error"]))
                        if verbose:
                            print(f"    [-] 失敗: {res['src']} (原因: {res['error']})", file=sys.stderr)

            elapsed = time.time() - start_time
            saved_bytes = combo_orig_size - combo_new_size
            saved_ratio = (saved_bytes / max(1, combo_orig_size)) * 100

            combo_stat = {
                "quality": q,
                "scale": s,
                "combo_name": combo_name,
                "output_dir": str(preset_out_dir),
                "total_files": total_files,
                "success_cnt": success_cnt,
                "error_cnt": error_cnt,
                "orig_size": combo_orig_size,
                "new_size": combo_new_size,
                "saved_bytes": saved_bytes,
                "saved_ratio": saved_ratio,
                "elapsed": elapsed,
                "errors": errors,
            }
            overall_results.append(combo_stat)

            if verbose:
                fps = success_cnt / max(0.001, elapsed)
                print(f"    [+] 該組合完成：耗時 {elapsed:.2f} 秒 ({fps:.1f} 張/秒)")
                print(f"        原始大小: {format_bytes(combo_orig_size)} -> 壓縮後: {format_bytes(combo_new_size)}")
                print(f"        節省空間: {format_bytes(saved_bytes)} ({saved_ratio:.2f}%)\n")

    # 4. 印出綜合統計報表
    if verbose and not dry_run and overall_results:
        print("==========================================================================================")
        print(" [📊] 批次圖片重新壓縮成果統計報表")
        print("==========================================================================================")
        print(f" {'組合名稱':<12} {'品質':<6} {'縮放':<8} {'原始總大小':<12} {'壓縮後總大小':<12} {'節省容量':<12} {'壓縮率':<10} {'耗時':<8}")
        print("-" * 90)
        for r in overall_results:
            scale_str = f"{int(r['scale']*100)}%"
            print(
                f" {r['combo_name']:<12} Q={r['quality']:<4} {scale_str:<8} "
                f"{format_bytes(r['orig_size']):<12} {format_bytes(r['new_size']):<12} "
                f"{format_bytes(r['saved_bytes']):<12} {r['saved_ratio']:>6.2f}%    "
                f"{r['elapsed']:>5.1f}s"
            )
        print("==========================================================================================")
        print(f"[🎉] 所有檔案均已完整維持子目錄階層鏡像輸出至：{out_dir}\n")

    return {
        "total_files": total_files,
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "presets": overall_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ScanORC: 批次圖片重新壓縮與縮放工具 (預設同時掃描 JPG 與 PNG，遞迴鏡像維持相對目錄結構)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用範例:
  # 1. 基本使用：壓縮指定目錄，品質設為 75，維持所有子目錄結構輸出至 output_q75:
  python recompress_images.py samples -o output_q75 -q 75

  # 2. 同時縮小尺寸 (例如寬高縮減至 75% 且品質 70，JPG 與 PNG 同步縮小與壓縮):
  python recompress_images.py samples -o output_compressed -q 70 -s 0.75

  # 3. 多組品質與縮放比例批次生成 (自動建立 q50_s100, q75_s75, q85_s50 等子目錄以供對比):
  python recompress_images.py samples -o output_compare -q 50 75 85 -s 1.0 0.75

  # 4. 指定最大長寬限制 (例如長邊不超過 1920 像素):
  python recompress_images.py samples -o output_hd -q 80 --max-dim 1920

  # 5. PNG 保持無損全彩 (僅縮放尺寸與 zlib 壓縮，不執行量化調色盤降階):
  python recompress_images.py samples -o output_lossless -q 80 -s 0.8 --png-lossless

  # 6. 模擬運行預覽清單 (不實際寫入任何檔案):
  python recompress_images.py samples -o output_preview -q 75 --dry-run

  # 7. 套用影像銳化遮色片 (增強文字與線條邊緣清晰度):
  python recompress_images.py samples -o output_sharp -q 75 -s 0.8 --unsharp 1x0.55+1.5+0.002
"""
    )

    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="輸入來源目錄路徑 (包含 JPG/PNG 圖檔與各層子目錄)"
    )
    parser.add_argument(
        "--input", "-i",
        dest="input_opt",
        default=None,
        help="輸入來源目錄路徑"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="輸出目標新目錄路徑 (將鏡像重建完整子目錄結構)"
    )
    parser.add_argument(
        "--quality", "-q",
        nargs="+",
        type=int,
        default=[50],
        help="壓縮品質 (1~100，預設: 50；支援傳入多個值如 -q 50 75 85。對 JPG 為編碼品質，對 PNG 為調色盤自適應量化品質)"
    )
    parser.add_argument(
        "--scale", "-s",
        nargs="+",
        default=["1.0"],
        help="圖片尺寸縮放比例 (例如 0.8 或 1.0，預設: 1.0；支援傳入多個值如 -s 1.0 0.8 0.5。同時作用於 JPG 與 PNG)"
    )
    parser.add_argument(
        "--max-dim", "-m",
        type=int,
        default=None,
        help="限制圖片最大長寬 (像素，例如 1920；若大於此值將等比例縮小)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="並行 Worker 數 (預設: 系統 CPU 核心數)"
    )
    parser.add_argument(
        "--png-lossless",
        action="store_true",
        help="強制 PNG 保持無損全彩 (不執行調色盤降階量化，僅套用 -s 縮放與 zlib 最佳化壓縮)"
    )
    parser.add_argument(
        "--subsampling",
        choices=["keep", "444", "422", "420"],
        default="keep",
        help="[JPG專用] 色彩抽樣模式 (keep: 保留原格式, 444: 無色度抽樣最清晰, 420: 壓縮率最高)"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="停用 Huffman / zlib 編碼表最佳化"
    )
    parser.add_argument(
        "--progressive",
        action="store_true",
        help="[JPG專用] 啟用漸進式 JPEG 編碼"
    )
    parser.add_argument(
        "--strip-metadata",
        action="store_true",
        help="移除 EXIF 與 DPI 元資料 (預設為保留)"
    )
    parser.add_argument(
        "--unsharp",
        default="1x0.55+1.5+0.002",
        help="影像銳化濾鏡參數 (ImageMagick unsharp 語法例如 1x0.55+1.5+0.002，預設: 1x0.55+1.5+0.002；若傳入 off/none 則關閉)"
    )
    parser.add_argument(
        "--no-unsharp",
        action="store_const",
        dest="unsharp",
        const=None,
        default=argparse.SUPPRESS,
        help="停用影像銳化濾鏡"
    )
    parser.add_argument(
        "--auto-grayscale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="自動偵測圖檔是否為灰階色調，若無彩色資訊則以灰階模式壓縮儲存以進一步節省空間 (預設: True，使用 --no-auto-grayscale 保持原樣)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="模擬執行模式 (僅列出檔案清單與目錄結構，不實際寫入)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="精簡輸出，僅顯示摘要與錯誤"
    )

    if attach_preference_and_template_support:
        args = attach_preference_and_template_support(parser, "recompress_images")
    else:
        args = parser.parse_args()

    in_target = args.input_opt or args.input
    if not in_target:
        print("[-] 錯誤：請指定輸入目錄！", file=sys.stderr)
        print("    使用方式: python recompress_images.py <輸入目錄> -o <輸出目錄> [選項]", file=sys.stderr)
        sys.exit(1)

    parsed_scales = [parse_scale_value(v) for v in args.scale]

    try:
        recompress_directory(
            input_dir=in_target,
            output_dir=args.output,
            qualities=args.quality,
            scales=parsed_scales,
            max_dim=args.max_dim,
            workers=args.workers,
            optimize=not args.no_optimize,
            progressive=args.progressive,
            subsampling=args.subsampling,
            preserve_metadata=not args.strip_metadata,
            png_lossless=args.png_lossless,
            unsharp=args.unsharp,
            auto_grayscale=args.auto_grayscale,
            dry_run=args.dry_run,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"\n[-] 錯誤: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
