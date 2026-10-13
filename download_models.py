#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

BASE_DIR = Path(__file__).resolve().parent
if "SCANORC_MODELS_DIR" in os.environ:
    DEFAULT_MODELS_DIR = os.environ["SCANORC_MODELS_DIR"]
elif (BASE_DIR / "models").exists():
    DEFAULT_MODELS_DIR = str((BASE_DIR / "models").resolve())
elif Path("/opt/models").exists():
    DEFAULT_MODELS_DIR = "/opt/models"
else:
    DEFAULT_MODELS_DIR = str((BASE_DIR / "models").resolve())


def main():
    parser = argparse.ArgumentParser(description="ScanORC: 下載 AI 模型權重 (DocLayout-YOLO 與 GLM-OCR)")
    parser.add_argument(
        "--models-dir", "-o",
        default=DEFAULT_MODELS_DIR,
        help=f"模型權重儲存母目錄 (預設: {DEFAULT_MODELS_DIR} 或 SCANORC_MODELS_DIR 環境變數)"
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    yolo_dir = models_dir / "doclayout_yolo"
    glm_dir = models_dir / "glm-ocr"

    yolo_dir.mkdir(parents=True, exist_ok=True)
    glm_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] 模型權重目標儲存目錄: {models_dir}")
    print("=== [1/2] 下載 DocLayout-YOLO 模型權重 ===")
    try:
        yolo_file = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
            local_dir=str(yolo_dir),
        )
        print(f"[+] DocLayout-YOLO 下載完成：{yolo_file}")
    except Exception as e:
        print(f"[-] DocLayout-YOLO 下載失敗：{e}", file=sys.stderr)
        sys.exit(1)

    print("\n=== [2/2] 下載 GLM-OCR 模型權重 (約 1.8GB) ===")
    try:
        glm_path = snapshot_download(
            repo_id="zai-org/GLM-OCR",
            local_dir=str(glm_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[+] GLM-OCR 下載完成：{glm_path}")
    except Exception as e:
        print(f"[-] GLM-OCR 下載失敗：{e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[🎉] 所有模型下載完畢！已存放於: {models_dir}")


if __name__ == "__main__":
    main()
