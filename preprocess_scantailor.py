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
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image

from config_utils import attach_preference_and_template_support

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


def convert_tif_to_target_format(
    tif_path: Path,
    target_path: Path,
    target_format: str = "jpg",
    quality: int = 95
):
    """
    將 ScanTailor 輸出的 TIF 轉換為指定的 JPG / PNG / TIF 格式
    """
    fmt = target_format.lower()
    if fmt in ("jpg", "jpeg"):
        with Image.open(tif_path) as im:
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            im.save(target_path, "JPEG", quality=quality)
    elif fmt == "png":
        with Image.open(tif_path) as im:
            im.save(target_path, "PNG", optimize=True)
    elif fmt in ("tif", "tiff"):
        shutil.copy2(tif_path, target_path)
    else:
        shutil.copy2(tif_path, target_path)


def export_image_to_target_format(
    src_path: Path,
    target_path: Path,
    target_format: str = "jpg",
    quality: int = 95
):
    """
    將原圖導出為指定的目標格式 (若格式相同直接快速複製，否則轉換儲存)
    """
    fmt = target_format.lower()
    target_ext = f".{fmt}" if fmt != "jpeg" else ".jpg"
    if src_path.suffix.lower() == target_ext and fmt != "jpg":
        shutil.copy2(src_path, target_path)
        return

    try:
        with Image.open(src_path) as im:
            if fmt in ("jpg", "jpeg"):
                if im.mode not in ("RGB", "L"):
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


def parse_skip_ocr_targets(skip_args: List[str]) -> set:
    """
    解析不作 ScanTailor 處理 / 略過 OCR 的頁面目標清單：
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
                print(f"[-] 讀取略過清單檔案失敗 '{item_str}': {e}", file=sys.stderr)
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
    判斷特定圖檔是否應略過 ScanTailor 處理：
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
            quality=args.quality
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
                    quality=args.quality
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
        default=85,
        help="若輸出為 JPG 時的壓縮品質 1-100 (預設: 85)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="單一子目錄內分批處理之頁數大小 (預設: 0 表示整目錄一次處理以保持章節尺寸統一；圖檔過多時可設 100~200)"
    )
    parser.add_argument(
        "--skip-orc", "--skip-ocr", "--no-ocr",
        nargs="*",
        default=[],
        dest="skip_ocr",
        help="略過 ScanTailor 幾何轉正與裁切處理的檔案清單 (可指定檔名、頁碼、路徑或 .txt 清單，直接將原始圖檔導出至輸出目錄)"
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
    print(f"[*] 輸出目錄: {output_base_dir}")
    print(f"[*] 參數設定: {args.format.upper()} (解析度: {dpi_summary}, 邊距: {args.margins}mm, 色彩: {args.color_mode})")
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
