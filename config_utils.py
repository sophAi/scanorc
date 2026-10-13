#!/usr/bin/env python3
"""
ScanORC: 統一 JSON 參數與偏好設定工具模組 (config_utils.py)
提供全專案所有工具程式一致的：
  1. 單一 JSON 檔案參數自動直通 (例如: python tool.py my_config.json)
  2. 讀取 JSON 偏好設定檔 (-p, --preference, -c, --config)
  3. 匯出完整參數預設值範本 (-t, --template, --dump-config)
  4. 支援帶註解的 JSON 檔案解析
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List


def strip_json_comments(text: str) -> str:
    """
    移除 JSON 字串中的單行註解 (// ...) 與多行區塊註解 (/* ... */)，
    同時保護字串常數內部的斜線內容不受影響。
    """
    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def load_json_with_comments(filepath_or_str: Any) -> Any:
    """
    讀取支援註解的 JSON 檔案或字串
    """
    path = Path(filepath_or_str)
    if path.is_file():
        content = path.read_text(encoding="utf-8", errors="replace")
    else:
        content = str(filepath_or_str)

    cleaned = strip_json_comments(content).strip()
    if not cleaned:
        return {}
    return json.loads(cleaned)


def extract_template_dict(parser: argparse.ArgumentParser) -> Dict[str, Any]:
    """
    從 ArgumentParser 的 _actions 列表中提取所有參數名稱與其預設值，組成完整的範本字典。
    """
    template = {}
    ignored_dests = {"help", "pref_file", "template_file"}

    for action in parser._actions:
        dest = action.dest
        if not dest or dest in ignored_dests:
            continue
        if action.default is argparse.SUPPRESS:
            continue

        default_val = action.default
        # 處理 None / 列表 / 基本型態
        template[dest] = default_val

    return template


def attach_preference_and_template_support(
    parser: argparse.ArgumentParser,
    tool_name: str = "",
    cli_args: Optional[List[str]] = None
) -> argparse.Namespace:
    """
    為傳入的 parser 註冊標準 -p / -t 旗標，支援單一 JSON 直通判定、載入設定與範本輸出。
    """
    raw_argv = list(cli_args if cli_args is not None else sys.argv[1:])

    # [步驟 1] 註冊通用旗標
    parser.add_argument(
        "-p", "--preference", "--config", "-c",
        dest="pref_file",
        default=None,
        metavar="FILE.json",
        help="載入 JSON 偏好設定檔並套用所有參數 (命令列參數可覆蓋之)"
    )
    parser.add_argument(
        "-t", "--template", "--dump-config",
        dest="template_file",
        nargs="?",
        const="",
        default=None,
        metavar="OUT.json",
        help="匯出包含所有參數與預設值之完整 JSON 範本檔 (可指定檔名，留空則輸出至終端機)"
    )

    # [步驟 2] 檢查是否觸發 -t / --template / --dump-config 匯出範本
    template_idx = -1
    for opt in ("-t", "--template", "--dump-config"):
        if opt in raw_argv:
            template_idx = raw_argv.index(opt)
            break

    if template_idx != -1:
        out_target = None
        # 檢查該 flag 後方是否接著一個非 flag 的檔案名稱
        if template_idx + 1 < len(raw_argv) and not raw_argv[template_idx + 1].startswith("-"):
            out_target = raw_argv[template_idx + 1]

        template_data = extract_template_dict(parser)
        formatted_json = json.dumps(template_data, ensure_ascii=False, indent=2)

        if out_target:
            out_path = Path(out_target).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(formatted_json + "\n", encoding="utf-8")
            print(f"[+] 已成功匯出包含所有參數之 JSON 範本檔: {out_path}")
        else:
            print(formatted_json)
        sys.exit(0)

    # [步驟 3] 智慧直通判斷：
    # 如果第一個位置參數是以 .json 結尾且未以 '-' 開頭
    # (例外: 若為 extract_toc 且該 json 是 TOC 結構資料檔而非 tool 參數檔則放行)
    transformed_argv = list(raw_argv)
    if (
        len(transformed_argv) >= 1
        and not transformed_argv[0].startswith("-")
        and transformed_argv[0].lower().endswith(".json")
    ):
        candidate_json = transformed_argv[0]
        is_tool_preference = True

        # 若是 extract_toc.py，檢查是否為 toc.json 資料檔
        if "extract_toc" in tool_name.lower() and Path(candidate_json).is_file():
            try:
                probe = load_json_with_comments(candidate_json)
                if isinstance(probe, dict) and "toc" in probe and "input" not in probe:
                    is_tool_preference = False
            except Exception:
                pass

        if is_tool_preference:
            transformed_argv = ["--preference", candidate_json] + transformed_argv[1:]

    # [步驟 4] 檢查是否有 --preference / -p 參數需要預先讀取
    pref_path_str = None
    for i, arg in enumerate(transformed_argv):
        if arg in ("-p", "--preference", "--config", "-c"):
            if i + 1 < len(transformed_argv):
                pref_path_str = transformed_argv[i + 1]
                break
        elif any(arg.startswith(f"{prefix}=") for prefix in ("--preference", "--config")):
            pref_path_str = arg.split("=", 1)[1]
            break

    if pref_path_str:
        pref_path = Path(pref_path_str).resolve()
        if not pref_path.is_file():
            print(f"[-] 錯誤：找不到指定的偏好設定檔 '{pref_path}'！", file=sys.stderr)
            sys.exit(1)

        try:
            data = load_json_with_comments(pref_path)
            if not isinstance(data, dict):
                print(f"[-] 錯誤：偏好設定檔 '{pref_path}' 內容必須為 JSON 字典物件！", file=sys.stderr)
                sys.exit(1)

            # 正規化鍵名 (例如 "skip-ocr" -> "skip_ocr")
            clean_data = {k.replace("-", "_"): v for k, v in data.items()}
            clean_data.pop("pref_file", None)
            clean_data.pop("template_file", None)

            # 類型自動適配 (例如 nargs='*' 時若給予單一字串自動轉為列表)
            action_map = {a.dest: a for a in parser._actions}
            for k, v in list(clean_data.items()):
                action = action_map.get(k)
                if action and action.nargs in ("*", "+") and isinstance(v, (str, int, float)):
                    clean_data[k] = [v]

            parser.set_defaults(**clean_data)
        except json.JSONDecodeError as e:
            print(f"[-] 錯誤：偏好設定檔 '{pref_path}' JSON 語法無效: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"[-] 錯誤：讀取偏好設定檔 '{pref_path}' 失敗: {e}", file=sys.stderr)
            sys.exit(1)

    # [步驟 5] 執行最終解析 (命令列參數享有最高優先級)
    return parser.parse_args(transformed_argv)
