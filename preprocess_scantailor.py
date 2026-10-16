#!/usr/bin/env python3
"""
ScanORC: 掃描書本前置預處理工具 (基於 ScanTailor 引擎)
功能：
  1. 遞迴掃描 --input 中的所有子目錄階層，並以完全對應的子目錄結構鏡像輸出至 --output
  2. 每個子目錄獨立判定：每掃描一個新的子目錄，視為一本獨立的書，使用全新的裁切與轉正機制，不沿用任何跨目錄狀態或快取
  3. 預設不作圖形縮放：保持 1:1 原生解析度與像素清晰度，僅執行傾斜轉正與多餘留白裁切；除非額外加上 --dpi 參數才變更圖形解析度
  4. 自動去除掃描儀壓痕黑邊與中縫漸層陰影
  5. 去除孤立雜訊斑點 (Despeckle)
  6. 轉出標準化高解析圖檔 (支援 JPG / PNG / TIF)，作為後續 OCR 與版面分析之最佳輸入源
"""

import os
import sys
import re
import time
import shutil
import fnmatch
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, ImageFilter

from config_utils import attach_preference_and_template_support
try:
    from recompress_images import is_grayscale_image
except ImportError:
    from PIL import ImageChops

    def is_grayscale_image(img_or_path, is_cover=False, **kwargs):
        if isinstance(img_or_path, (str, Path)):
            try:
                with Image.open(img_or_path) as im:
                    return is_grayscale_image(im, is_cover=is_cover, **kwargs)
            except Exception:
                return False
        im = img_or_path
        if is_cover is None:
            is_cover = False
        if im.mode in ("L", "1"):
            return True
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        r, g, b = im.split()[:3]
        d_rg = ImageChops.difference(r, g)
        d_gb = ImageChops.difference(g, b)
        d_rb = ImageChops.difference(r, b)
        max_diff = max(d_rg.getextrema()[1], d_gb.getextrema()[1], d_rb.getextrema()[1])
        if max_diff == 0:
            return True
        if is_cover:
            return False
        if max_diff <= 5:
            return True
        h_rg, h_gb, h_rb = d_rg.histogram(), d_gb.histogram(), d_rb.histogram()
        c_high = max(sum(h_rg[25:]), sum(h_gb[25:]), sum(h_rb[25:]))
        if c_high > 15:
            return False
        c_mid = max(sum(h_rg[12:]), sum(h_gb[12:]), sum(h_rb[12:]))
        if c_mid > 40:
            return False
        return True

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def check_scantailor_cmd() -> List[str]:
    """
    檢測 ScanTailor-CLI 可用指令：
    1. 最優先：專案內置免安裝 Portable 版 (bin/scantailor-cli，方案 A - 作法 1)
    2. 次優先：系統 PATH 中的原生 scantailor-cli
    3. 備援：Flatpak 沙盒安裝的 ScanTailor
    """
    # 1. 最優先檢查專案內置的獨立可攜式 scantailor-cli (方案 A - 作法 1)
    base_dir = Path(__file__).parent.resolve()
    bundled_cli = base_dir / "bin" / "scantailor-cli"
    if bundled_cli.is_file() and os.access(bundled_cli, os.X_OK):
        return [str(bundled_cli)]

    # 2. 檢查系統 PATH 中的原生 scantailor-cli
    native_cli = shutil.which("scantailor-cli")
    if native_cli:
        return [native_cli]

    # 3. 檢查 Flatpak 安裝的 ScanTailor
    flatpak_bin = shutil.which("flatpak")
    if flatpak_bin:
        res = subprocess.run(
            [flatpak_bin, "run", "--command=scantailor-cli", "org.scantailor.ScanTailor", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if res.returncode == 0:
            return [flatpak_bin, "run", "--command=scantailor-cli", "org.scantailor.ScanTailor"]

    return []


def get_directory_dpi(img_files: List[Path], fallback_dpi: int = 300) -> int:
    """
    從子目錄圖片中檢測原始掃描 DPI (透過圖片 Header/EXIF 元數據)
    若圖片未標記 DPI，則回傳 fallback_dpi
    """
    for img_path in img_files[:5]:
        try:
            with Image.open(img_path) as im:
                dpi = im.info.get("dpi")
                if dpi and isinstance(dpi, tuple) and dpi[0] > 0:
                    return int(round(dpi[0]))
        except Exception:
            continue
    return fallback_dpi


def apply_image_transform(
    src_path: Path,
    target_path: Path,
    target_format: str = "jpg",
    quality: int = 50,
    scale: float = 1.0,
    unsharp: Optional[str] = "1x0.55+1.5+0.002",
    auto_grayscale: bool = True
):
    """
    通用影像導出與處理函式：
    支援尺寸縮放 (scale)、銳化遮色片 (unsharp，預設 1x0.55+1.5+0.002)、
    JPG 品質壓縮 (quality) 與自動灰階判定 (auto_grayscale，保護封面與彩色細節)。
    優先使用 ImageMagick convert 執行高效精確之 -unsharp 與 -colorspace Gray 濾鏡，若無則自動降級為 PIL UnsharpMask。
    """
    fmt = target_format.lower()
    target_ext = f".{fmt}" if fmt != "jpeg" else ".jpg"
    is_same_ext = (src_path.suffix.lower() == target_ext)
    no_scale = (scale is None or abs(scale - 1.0) <= 1e-4)
    has_unsharp = bool(unsharp and str(unsharp).strip().lower() not in ("none", "off", "0", "false"))

    # 若無需轉換格式、縮放或銳化，且非 JPG 格式，直接快速複製
    if is_same_ext and fmt not in ("jpg", "jpeg") and no_scale and not has_unsharp:
        shutil.copy2(src_path, target_path)
        return

    # 灰階檢測 (針對 JPG 輸出)
    is_gray = False
    if auto_grayscale and fmt in ("jpg", "jpeg"):
        try:
            name_lower = target_path.name.lower()
            src_name_lower = src_path.name.lower()
            is_cover_target = bool(
                "cover" in name_lower
                or "cover" in src_name_lower
                or "backcover" in name_lower
                or "backcover" in src_name_lower
            )
            is_gray = is_grayscale_image(src_path, is_cover=is_cover_target)
        except Exception:
            is_gray = False

    # 1. 優先嘗試 ImageMagick (convert / magick)
    convert_bin = shutil.which("convert") or shutil.which("magick")
    if convert_bin:
        cmd = [convert_bin, str(src_path)]
        if not no_scale and scale > 0:
            scale_pct = round(scale * 100, 2)
            cmd.extend(["-resize", f"{scale_pct}%"])
        if has_unsharp:
            cmd.extend(["-unsharp", str(unsharp).strip()])
        if fmt in ("jpg", "jpeg"):
            if is_gray:
                cmd.extend(["-colorspace", "Gray"])
            cmd.extend(["-quality", str(quality)])
        cmd.append(str(target_path))

        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and target_path.exists():
                return
        except Exception:
            pass

    # 2. 備援方案：PIL (Pillow)
    try:
        with Image.open(src_path) as im:
            if not no_scale and scale > 0:
                new_w = max(1, int(round(im.width * scale)))
                new_h = max(1, int(round(im.height * scale)))
                im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)

            if has_unsharp:
                # 解析 radiusxsigma+amount+threshold (例如 1x0.55+1.5+0.002)
                r_val, p_val, th_val = 1.0, 150, 0
                m = re.match(r"^(\d*(?:\.\d+)?)x(\d*(?:\.\d+)?)\+([0-9.]+)\+([0-9.]+)$", str(unsharp).strip())
                if m:
                    rad_s, sig_s, amt_s, th_s = m.groups()
                    rad = float(rad_s) if rad_s else 1.0
                    sig = float(sig_s) if sig_s else 0.55
                    amt = float(amt_s) if amt_s else 1.5
                    th = float(th_s) if th_s else 0.002
                    r_val = rad if rad > 0 else sig
                    p_val = int(round(amt * 100))
                    th_val = int(round(th * 255))
                im = im.filter(ImageFilter.UnsharpMask(radius=r_val, percent=p_val, threshold=th_val))

            if fmt in ("jpg", "jpeg"):
                if is_gray:
                    im = im.convert("L")
                elif im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                im.save(target_path, "JPEG", quality=quality)
            elif fmt == "png":
                im.save(target_path, "PNG", optimize=True)
            elif fmt in ("tif", "tiff"):
                im.save(target_path, "TIFF")
            else:
                shutil.copy2(src_path, target_path)
    except Exception:
        shutil.copy2(src_path, target_path)


def convert_tif_to_target_format(
    tif_path: Path,
    target_path: Path,
    target_format: str = "jpg",
    quality: int = 50,
    scale: float = 1.0,
    unsharp: Optional[str] = "1x0.55+1.5+0.002",
    auto_grayscale: bool = True
):
    """
    將 ScanTailor 輸出的 TIF 轉換為指定的 JPG / PNG / TIF 格式，並支援尺寸縮放與銳化
    """
    apply_image_transform(
        src_path=tif_path,
        target_path=target_path,
        target_format=target_format,
        quality=quality,
        scale=scale,
        unsharp=unsharp,
        auto_grayscale=auto_grayscale
    )


def export_image_to_target_format(
    src_path: Path,
    target_path: Path,
    target_format: str = "jpg",
    quality: int = 50,
    scale: float = 1.0,
    unsharp: Optional[str] = "1x0.55+1.5+0.002",
    auto_grayscale: bool = True
):
    """
    將原圖導出為指定的目標格式 (支援尺寸縮放與銳化)
    """
    apply_image_transform(
        src_path=src_path,
        target_path=target_path,
        target_format=target_format,
        quality=quality,
        scale=scale,
        unsharp=unsharp,
        auto_grayscale=auto_grayscale
    )


def parse_skip_ocr_targets(skip_args: List[str]) -> set:
    """
    解析不作 ScanTailor 處理 / 略過 OCR 的頁面目標清單：
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
                print(f"[-] 讀取略過清單檔案失敗 '{item_str}': {e}", file=sys.stderr)
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
    判斷特定圖檔是否應略過 ScanTailor 處理：
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


def collect_image_groups(input_paths: List[Path], output_base_dir: Optional[Path] = None) -> List[Tuple[Path, List[Path]]]:
    """
    掃描輸入路徑清單 (支援傳入目錄與單個/多個檔案)，按子目錄分組收集所有支援的圖檔：
    傳回值為清單：[(相對目錄路徑 rel_dir, [該目錄下的圖檔路徑清單]), ...]
    """
    groups = []
    standalone_files = []

    for p in input_paths:
        p = p.resolve()
        if p.is_file():
            if p.suffix.lower() in VALID_IMAGE_EXTS:
                standalone_files.append(p)
            else:
                print(f"[-] 警告：跳過不支援的格式 '{p.name}'", file=sys.stderr)
        elif p.is_dir():
            for root, dirs, files in os.walk(p):
                # 防止輸出目錄位於輸入目錄內部時發生遞迴自包含掃描
                if output_base_dir:
                    dirs[:] = [
                        d for d in dirs
                        if not (Path(root) / d).resolve() == output_base_dir
                        and not (Path(root) / d).resolve().is_relative_to(output_base_dir)
                    ]
                dirs.sort()
                img_files = [
                    Path(root) / f for f in sorted(files)
                    if Path(f).suffix.lower() in VALID_IMAGE_EXTS
                ]
                if img_files:
                    rel_dir = Path(root).relative_to(p)
                    groups.append((rel_dir, img_files))
        else:
            print(f"[-] 警告：輸入路徑 '{p}' 不存在，略過！", file=sys.stderr)

    if standalone_files:
        groups.insert(0, (Path("."), sorted(standalone_files)))

    return groups


def process_image_group(
    rel_dir: Path,
    img_files: List[Path],
    output_base_dir: Path,
    scantailor_cmd: List[str],
    args: argparse.Namespace,
    skip_targets: set
) -> int:
    """
    針對每個獨立子目錄 (對應一本書的掃描檔)：
    1. 使用全新、獨立的轉正與裁切計算，不共享任何全域狀態或歷史目錄規格
    2. 預設不作圖形縮放 (input_dpi == output_dpi)，保持 1:1 原生解析度；僅在指定 --dpi 時進行縮放
    3. 若圖檔位於略過清單 (skip_targets) 中，直接繞過 ScanTailor 運算，無損導出至目標目錄
    """
    dest_dir = output_base_dir / rel_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 分流：篩選出略過圖檔與需前置處理之圖檔
    files_to_skip = [p for p in img_files if should_skip_ocr(p, skip_targets)]
    files_to_process = [p for p in img_files if not should_skip_ocr(p, skip_targets)]

    total_converted = 0

    # 1. 處理略過圖檔：直接導出原始圖檔至目標目錄，繞過 ScanTailor
    for skip_img in files_to_skip:
        target_name = f"{skip_img.stem}.{args.format.lower()}"
        final_dest = dest_dir / target_name
        export_image_to_target_format(
            skip_img,
            final_dest,
            target_format=args.format,
            quality=args.quality,
            scale=args.scale,
            unsharp=args.unsharp,
            auto_grayscale=args.auto_grayscale
        )
        total_converted += 1
        print(f"    [*] [略過 ScanTailor 處理] {skip_img.name} -> 直接導出原始圖檔至目標目錄")

    # 若所有圖檔皆被略過，直接返回
    if not files_to_process:
        return total_converted

    # 建立該子目錄專屬的臨時工作區，執行前徹底清空確保無殘留
    temp_tif_dir = dest_dir / ".scantailor_temp"
    shutil.rmtree(temp_tif_dir, ignore_errors=True)
    temp_tif_dir.mkdir(parents=True, exist_ok=True)

    # 檢測待處理圖檔的原生 DPI
    detected_dpi = get_directory_dpi(files_to_process)

    # 解析度處理邏輯：
    # 若使用者未傳入 --dpi (預設 None)，設定 in_dpi = out_dpi = detected_dpi，比率 1:1，不縮放圖形
    # 若使用者傳入 --dpi，則設定 out_dpi 為使用者指定值，進行重新取樣
    if args.dpi is not None and args.dpi > 0:
        in_dpi = detected_dpi
        out_dpi = args.dpi
        scaling_desc = f"自訂輸出解析度 {out_dpi} DPI (將執行圖形重新取樣)"
    else:
        in_dpi = detected_dpi
        out_dpi = detected_dpi
        scaling_desc = f"1:1 原生尺寸無損輸出 (原生 {in_dpi} DPI，不作圖形縮放)"

    print(f"    [*] 運算機制: 獨立書籍版面判定 (獨立計算傾斜角與邊界留白)")
    print(f"    [*] 解析度策略: {scaling_desc}")

    # 判斷是否需要按批次分批送給 ScanTailor
    batch_size = args.batch_size
    if batch_size > 0 and len(files_to_process) > batch_size:
        batches = [files_to_process[i:i + batch_size] for i in range(0, len(files_to_process), batch_size)]
    else:
        batches = [files_to_process]

    try:
        for batch in batches:
            base_opts = [
                "--layout=1",                 # 單頁版面
                "--deskew=auto",              # 自動傾斜角度校正 (各目錄獨立計算)
                "--content-detection=normal", # 智慧內容邊界識別 (各目錄獨立判斷)
                f"--margins={args.margins}",  # 版心頁邊距 (mm)
                f"--color-mode={args.color_mode}", # 保留插圖階調
                f"--dpi={in_dpi}",            # 輸入解析度
                f"--output-dpi={out_dpi}",    # 輸出解析度 (兩者相等時比例為 1.0，完全不縮放)
                f"--despeckle={args.despeckle}", # 消除黑斑雜點
            ]
            cmd = (
                scantailor_cmd
                + base_opts
                + [str(p.resolve()) for p in batch]
                + [str(temp_tif_dir.resolve())]
            )

            # 執行 ScanTailor 運算
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # 收集產生的 TIF 檔案並轉出至目標鏡像目錄
            tif_files = sorted(
                list(temp_tif_dir.glob("*.tif")) + list(temp_tif_dir.glob("*.tiff"))
            )

            for tif_file in tif_files:
                target_name = f"{tif_file.stem}.{args.format.lower()}"
                final_dest = dest_dir / target_name
                convert_tif_to_target_format(
                    tif_file,
                    final_dest,
                    target_format=args.format,
                    quality=args.quality,
                    scale=args.scale,
                    unsharp=args.unsharp,
                    auto_grayscale=args.auto_grayscale
                )
                total_converted += 1

            # 每批完成後清理中間產物，節省硬碟空間
            for f in tif_files:
                try:
                    f.unlink()
                except OSError:
                    pass
            shutil.rmtree(temp_tif_dir / "cache", ignore_errors=True)

    finally:
        # 確保刪除暫存工作目錄
        shutil.rmtree(temp_tif_dir, ignore_errors=True)

    return total_converted


def main():
    parser = argparse.ArgumentParser(
        description="ScanORC Preprocess: 掃描書本轉正、去邊與版心標準化前置處理工具 (各目錄獨立判斷 / 預設不縮放)"
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        default=["samples"],
        help="輸入來源：待處理的圖片檔案路徑或目錄 (支援多個檔案、萬用字元或目錄，預設: samples)"
    )
    parser.add_argument(
        "--output", "-o",
        default="preprocessed_samples",
        help="輸出目錄：儲存處理後圖檔的目錄 (自動建立並鏡像原始子目錄階層，預設: preprocessed_samples)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["jpg", "png", "tif"],
        default="jpg",
        help="最終輸出圖檔格式 (預設: jpg)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="輸出解析度 DPI (預設: None，表示不作圖形縮放，保持 1:1 原始解析度；僅在明確指定此參數時才變更解析度)"
    )
    parser.add_argument(
        "--margins",
        type=int,
        default=1,
        help="版心四周均勻邊距大小 (預設: 1 mm)"
    )
    parser.add_argument(
        "--color-mode",
        choices=["color_grayscale", "mixed", "black_and_white"],
        default="color_grayscale",
        help="色彩模式：color_grayscale (保留照片/圖表階調)、mixed (圖文混合)、black_and_white (純黑白)"
    )
    parser.add_argument(
        "--despeckle",
        choices=["off", "cautious", "normal", "aggressive"],
        default="normal",
        help="去雜訊去斑強度 (預設: normal)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=50,
        help="若輸出為 JPG 時的壓縮品質 1-100 (預設: 50)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="輸出圖檔尺寸縮放比例 (例如 0.8 或 1.0，預設: 1.0)"
    )
    parser.add_argument(
        "--unsharp",
        default="1x0.55+1.5+0.002",
        help="影像銳化濾鏡參數 (ImageMagick unsharp 語法，例如 1x0.55+1.5+0.002，設為 none 或 off 則關閉，預設: 1x0.55+1.5+0.002)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="單一子目錄內分批處理之頁數大小 (預設: 0 表示整目錄一次處理以保持章節尺寸統一；圖檔過多時可設 100~200)"
    )
    parser.add_argument(
        "--auto-grayscale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="自動偵測圖檔是否為灰階色調，若無彩色資訊則改以灰階模式壓縮儲存以節省空間 (預設: True，使用 --no-auto-grayscale 保持原樣)"
    )
    parser.add_argument(
        "--skip-orc", "--skip-ocr", "--no-ocr",
        nargs="*",
        default=[],
        dest="skip_ocr",
        help="略過 ScanTailor 幾何轉正與裁切處理的檔案清單 (支援萬用字元如 Cover-*.jpg、Content-*、檔名、頁碼或 txt 清單，直接將原始圖檔導出至輸出目錄)"
    )
    args = attach_preference_and_template_support(parser, "preprocess_scantailor")

    scantailor_cmd = check_scantailor_cmd()
    if not scantailor_cmd:
        print("[-] 錯誤：未偵測到 scantailor-cli！", file=sys.stderr)
        print("    請確認專案 bin/scantailor-cli 是否存在，或安裝系統版 ScanTailor", file=sys.stderr)
        sys.exit(1)

    cmd_type = "專案內置免安裝 Portable 版" if "bin/scantailor-cli" in scantailor_cmd[0] else ("Flatpak 沙盒版" if "flatpak" in scantailor_cmd[0] else "系統原生版")

    input_list = args.input if isinstance(args.input, list) else [args.input]
    input_paths = [Path(p).resolve() for p in input_list]
    output_base_dir = Path(args.output).resolve()
    output_base_dir.mkdir(parents=True, exist_ok=True)

    image_groups = collect_image_groups(input_paths, output_base_dir)
    total_images = sum(len(imgs) for _, imgs in image_groups)

    if not image_groups or total_images == 0:
        print(f"[-] 在輸入清單 {args.input} 中未找到任何支援的圖片檔案！", file=sys.stderr)
        sys.exit(1)

    skip_targets = parse_skip_ocr_targets(args.skip_ocr)

    dpi_summary = f"{args.dpi} DPI (變更解析度)" if args.dpi else "原生 1:1 不縮放"

    print("==================================================")
    print(" ScanORC 掃描影像前置轉正與去邊處理系統")
    print(" 核心引擎: ScanTailor-CLI (獨立子目錄判斷 / 預設不縮放)")
    print("==================================================")
    print(f"[*] 執行引擎: {scantailor_cmd[0]} ({cmd_type})")
    print(f"[*] 輸入來源: {args.input}")
    quality_desc = f", 品質: {args.quality}" if args.format.lower() in ("jpg", "jpeg") else ""
    gray_desc = f", 自動灰階: {'啟用' if args.auto_grayscale else '關閉'}" if args.format.lower() in ("jpg", "jpeg") else ""
    print(f"[*] 參數設定: {args.format.upper()} (解析度: {dpi_summary}, 邊距: {args.margins}mm, 色彩: {args.color_mode}{quality_desc}{gray_desc}, 縮放: {args.scale}, 銳化: {args.unsharp})")
    if skip_targets:
        print(f"[*] 略過設定: 發現 {len(skip_targets)} 個略過項目 (將繞過 ScanTailor 運算並直接導出原圖)")
    print(f"[*] 掃描統計: 發現 {len(image_groups)} 個書籍/章節目錄群組，共計 {total_images} 張圖片待處理")
    print("--------------------------------------------------")

    for idx, (rel_dir, imgs) in enumerate(image_groups, 1):
        dir_display = str(rel_dir) if str(rel_dir) != "." else "[根目錄]"
        print(f"  ({idx}/{len(image_groups)}) 目錄: {dir_display} ({len(imgs)} 張圖片)")
    print("==================================================")

    start_time = time.time()
    grand_total_converted = 0

    for idx, (rel_dir, imgs) in enumerate(image_groups, 1):
        dir_display = str(rel_dir) if str(rel_dir) != "." else "[根目錄]"
        target_dir = output_base_dir / rel_dir
        print(f"\n[*] [{idx}/{len(image_groups)}] 正在處理子目錄: {dir_display} ({len(imgs)} 頁)...")
        print(f"    目標位置: {target_dir}")

        group_start = time.time()
        try:
            converted = process_image_group(
                rel_dir=rel_dir,
                img_files=imgs,
                output_base_dir=output_base_dir,
                scantailor_cmd=scantailor_cmd,
                args=args,
                skip_targets=skip_targets
            )
            grand_total_converted += converted
            group_elapsed = time.time() - group_start
            print(f"    [+] 完成！成功處理 {converted} 頁 (耗時: {group_elapsed:.1f} 秒)")
        except Exception as e:
            print(f"    [-] 處理目錄 '{dir_display}' 時發生錯誤: {e}", file=sys.stderr)

    total_elapsed = time.time() - start_time
    print("\n==================================================")
    print(f"[🎉] 所有影像前置轉正與去邊處理完成！")
    print(f"     總成功處理: {grand_total_converted} / {total_images} 頁")
    print(f"     總執行耗時: {total_elapsed:.1f} 秒")
    print(f"     鏡像輸出於: {output_base_dir}")
    print("==================================================")
    print(f"[提示] 若需進行 OCR 辨識，可直接指定處理後的目錄：")
    print(f"       python pipeline.py --samples {args.output} --output output --epub\n")


if __name__ == "__main__":
    main()
