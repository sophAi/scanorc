import os
import sys
from huggingface_hub import hf_hub_download, snapshot_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
YOLO_DIR = os.path.join(MODELS_DIR, "doclayout_yolo")
GLM_DIR = os.path.join(MODELS_DIR, "glm-ocr")

os.makedirs(YOLO_DIR, exist_ok=True)
os.makedirs(GLM_DIR, exist_ok=True)

print("=== [1/2] 下載 DocLayout-YOLO 模型權重 ===")
try:
    yolo_file = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        local_dir=YOLO_DIR,
    )
    print(f"DocLayout-YOLO 下載完成：{yolo_file}")
except Exception as e:
    print(f"DocLayout-YOLO 下載失敗：{e}", file=sys.stderr)
    sys.exit(1)

print("\n=== [2/2] 下載 GLM-OCR 模型權重 (約 1.8GB) ===")
try:
    glm_path = snapshot_download(
        repo_id="zai-org/GLM-OCR",
        local_dir=GLM_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"GLM-OCR 下載完成：{glm_path}")
except Exception as e:
    print(f"GLM-OCR 下載失敗：{e}", file=sys.stderr)
    sys.exit(1)

print("\n=== 所有模型下載完畢！===")
