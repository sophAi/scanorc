#!/usr/bin/env python3
"""
ScanORC: 視覺化圖形控制台 (app_gui.py)
基於 PySide6 實作，提供 5 大步驟 + 1 鍵總控流程：
  - 步驟 1: 幾何轉正與裁切 (preprocess_scantailor.py)
  - 步驟 2: 目錄結構抽取 (extract_toc.py)
  - 步驟 3: AI 版面分析與 OCR + HTML/EPUB 生成 (pipeline.py)
  - 步驟 4: 章節目錄歸檔 (extract_toc.py --organize-pages)
  - 步驟 5: 頁面微創替換與快速重編 (patch_pages.py)
  - 總控: 一鍵端到端自動化處理 (process_book.py)

底層完全解耦，透過 QProcess 非同步呼叫獨立的 Python 腳本，
所有終端機命令維持 100% 獨立可用。
"""

import os
import sys
import re
import json
import shutil
import subprocess
import shlex
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime

from config_utils import load_json_with_comments

# -------------------------------------------------------------
# Linux / Conda 環境相容性自動引導 (途徑 C: PyQt6 原生輸入法優先適配 + PySide6 雙引擎)
# -------------------------------------------------------------
project_root = Path(__file__).resolve().parent
bundled_lib_dir = project_root / "bin" / "lib"

# 1. 在 Linux 環境下，優先檢查是否可切換至相容系統原生 Fcitx5 輸入法外掛的 Python3 (PyQt6)
if __name__ == "__main__" and sys.platform.startswith("linux") and not os.environ.get("_SCANORC_GUI_BOOTSTRAPPED"):
    can_use_current_pyqt6 = False
    try:
        import PyQt6
        can_use_current_pyqt6 = True
    except ImportError:
        pass

    sys_py = "/usr/bin/python3"
    current_script = str(Path(__file__).resolve())
    if not can_use_current_pyqt6 and Path(sys_py).exists() and sys.executable != sys_py:
        try:
            chk = subprocess.run([sys_py, "-c", "import PyQt6.QtWidgets"], capture_output=True)
            if chk.returncode == 0:
                os.environ["_SCANORC_GUI_BOOTSTRAPPED"] = "1"
                os.environ["_SCANORC_ORIGINAL_PYTHON"] = sys.executable
                os.execv(sys_py, [sys_py, current_script] + sys.argv[1:])
        except Exception:
            pass

    # 若維持在當前 Python，檢查是否需要透過 LD_PRELOAD 解決 Conda 舊版 libxkbcommon 造成的段錯誤
    preload_candidates = [
        bundled_lib_dir / "libxkbcommon-x11.so.0",
        bundled_lib_dir / "libxkbcommon.so.0",
        Path("/usr/lib/x86_64-linux-gnu/libxkbcommon-x11.so.0"),
        Path("/usr/lib/x86_64-linux-gnu/libxkbcommon.so.0"),
    ]
    preload_libs = [str(p) for p in preload_candidates if p.exists()]
    seen = set()
    unique_libs = [x for x in preload_libs if not (x in seen or seen.add(x))]

    current_preload = os.environ.get("LD_PRELOAD", "")
    if unique_libs and "libxkbcommon-x11" not in current_preload:
        os.environ["_SCANORC_GUI_BOOTSTRAPPED"] = "1"
        os.environ["LD_PRELOAD"] = ":".join(unique_libs) + (f":{current_preload}" if current_preload else "")

        try:
            import PySide6
            qt_plugins_dir = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins"
            if qt_plugins_dir.exists():
                os.environ["QT_PLUGIN_PATH"] = str(qt_plugins_dir)
        except Exception:
            pass

        # 自身重啟以套用更新後的動態鏈接環境
        os.execv(sys.executable, [sys.executable, current_script] + sys.argv[1:])

# 2. 雙引擎 Qt 導入層 (PyQt6 優先，PySide6 作為 fallback)
try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import Qt, QProcess, pyqtSignal as Signal, pyqtSlot as Slot
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QSplitter, QTabWidget, QLabel, QLineEdit, QPushButton,
        QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
        QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QFileDialog, QMessageBox,
        QGroupBox, QFormLayout, QProgressBar, QFrame
    )

    # 確保全域輸入法模組使用 fcitx 原生通道
    if sys.platform.startswith("linux") and "QT_IM_MODULE" not in os.environ:
        os.environ["QT_IM_MODULE"] = "fcitx"

    # PyQt6 全域列舉向後相容對齊：將 Scoped Enums 回填至主類別命名空間，確保與 PySide6 / 舊式語法 100% 雙向相容
    def _patch_qt6_enums():
        for mod in (QtCore, QtGui, QtWidgets):
            for cls_name in dir(mod):
                if cls_name.startswith("_"):
                    continue
                try:
                    cls = getattr(mod, cls_name)
                    if isinstance(cls, type):
                        for sub_name in dir(cls):
                            if sub_name.startswith("_"):
                                continue
                            try:
                                sub = getattr(cls, sub_name)
                                if isinstance(sub, type):
                                    for val_name in dir(sub):
                                        if not val_name.startswith("_") and not hasattr(cls, val_name):
                                            try:
                                                setattr(cls, val_name, getattr(sub, val_name))
                                            except Exception:
                                                pass
                            except Exception:
                                pass
                except Exception:
                    pass
        # 常用複合對齊旗標
        setattr(QtCore.Qt, "AlignCenter", QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)

    _patch_qt6_enums()
    QT_LIB = "PyQt6"

except ImportError:
    # Fallback 至 PySide6
    try:
        import PySide6
        # 強制設定 QT_PLUGIN_PATH 指向 PySide6 自帶的外掛目錄
        pyside_dir = Path(PySide6.__file__).resolve().parent
        qt_plugins_dir = pyside_dir / "Qt" / "plugins"
        if qt_plugins_dir.exists():
            os.environ["QT_PLUGIN_PATH"] = str(qt_plugins_dir)

        # 自動補充缺失之庫
        qt_lib_dir = pyside_dir / "Qt" / "lib"
        if qt_lib_dir.exists() and bundled_lib_dir.exists() and os.access(qt_lib_dir, os.W_OK):
            for fn in [
                "libxcb-cursor.so.0", "libxcb-cursor.so.0.0.0",
                "libxkbcommon.so.0", "libxkbcommon.so.0.0.0",
                "libxkbcommon-x11.so.0", "libxkbcommon-x11.so.0.0.0"
            ]:
                src = bundled_lib_dir / fn
                dst = qt_lib_dir / fn
                if src.exists() and not dst.exists():
                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        pass
    except ImportError:
        pass

    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt, QProcess, Signal, Slot
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QSplitter, QTabWidget, QLabel, QLineEdit, QPushButton,
        QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
        QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QFileDialog, QMessageBox,
        QGroupBox, QFormLayout, QProgressBar, QFrame
    )
    QT_LIB = "PySide6"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def natural_sort_key(s: str) -> List:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def strip_ansi(text: str) -> str:
    """去除終端機 ANSI 色彩跳脫碼"""
    ansi_regex = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_regex.sub("", text)


def format_rel_path(p: Union[str, Path]) -> str:
    """將路徑格式化為相對於目前目錄的簡潔路徑 (若在外則保留絕對路徑)"""
    path = Path(p).resolve()
    try:
        rel = path.relative_to(Path.cwd().resolve())
        return str(rel)
    except ValueError:
        return str(path)


def format_paths_arg(paths: List[Union[str, Path]]) -> str:
    """將多個檔案路徑格式化為 CLI 參數字串 (含空白自動加引號)"""
    res = []
    for p in paths:
        s = format_rel_path(p)
        if " " in s:
            res.append(f'"{s}"')
        else:
            res.append(s)
    return " ".join(res)


def safe_split_args(cmd_str: str) -> List[str]:
    """安全解析可能包含引號的命令參數字串"""
    if not cmd_str.strip():
        return []
    try:
        return shlex.split(cmd_str)
    except ValueError:
        return cmd_str.strip().split()


# -------------------------------------------------------------
# 終端機色彩主題與語法高亮定義
# -------------------------------------------------------------
TERMINAL_THEMES = {
    "dark": {
        "name": "🌌 現代深灰 (Tokyo Night)",
        "bg": "#18181b",
        "fg": "#e4e4e7",
        "selection_bg": "#27272a",
        "info": "#38bdf8",     # [*] 天藍
        "success": "#4ade80",  # [+] 嫩綠
        "warning": "#fbbf24",  # [!] 金黃
        "error": "#f87171",    # [-] 珊瑚紅
        "dim": "#71717a",      # === 灰暗
    },
    "matrix": {
        "name": "📟 經典駭客綠 (Matrix Green)",
        "bg": "#0c100c",
        "fg": "#00ff66",
        "selection_bg": "#143a14",
        "info": "#66ffcc",
        "success": "#00ff66",
        "warning": "#ffff66",
        "error": "#ff3333",
        "dim": "#008833",
    },
    "solarized": {
        "name": "🌊 經典深藍 (Solarized Dark)",
        "bg": "#002b36",
        "fg": "#93a1a1",
        "selection_bg": "#073642",
        "info": "#268bd2",
        "success": "#859900",
        "warning": "#b58900",
        "error": "#dc322f",
        "dim": "#586e75",
    },
    "amber": {
        "name": "📻 復古琥珀橘 (Amber CRT)",
        "bg": "#140c00",
        "fg": "#ffb000",
        "selection_bg": "#331a00",
        "info": "#ffd24d",
        "success": "#ffb000",
        "warning": "#ffe066",
        "error": "#ff4d4d",
        "dim": "#805800",
    },
    "light": {
        "name": "📜 柔和白底 (Paper Light)",
        "bg": "#fafafa",
        "fg": "#24292f",
        "selection_bg": "#c8e1ff",
        "info": "#0969da",
        "success": "#1a7f37",
        "warning": "#9a6700",
        "error": "#cf222e",
        "dim": "#8c959f",
    },
}


class TerminalTextEdit(QTextEdit):
    """支援 Ctrl + 滑鼠滾輪即時縮放字型的終端機文字元件"""
    zoom_requested = Signal(int)  # +1: 放大, -1: 縮小

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_requested.emit(1)
            elif delta < 0:
                self.zoom_requested.emit(-1)
            event.accept()
        else:
            super().wheelEvent(event)


class TerminalLogHighlighter(QtGui.QSyntaxHighlighter):
    """終端機日誌即時語法與狀態高亮著色器"""
    def __init__(self, parent_document, theme_key="dark"):
        super().__init__(parent_document)
        self.theme_key = theme_key
        self.rules = []
        self.init_rules()

    def set_theme(self, theme_key: str):
        self.theme_key = theme_key
        self.init_rules()
        self.rehighlight()

    def init_rules(self):
        theme = TERMINAL_THEMES.get(self.theme_key, TERMINAL_THEMES["dark"])
        self.rules = []

        # 1. 成功完成標記 [+]
        fmt_success = QtGui.QTextCharFormat()
        fmt_success.setForeground(QtGui.QColor(theme["success"]))
        fmt_success.setFontWeight(QtGui.QFont.Weight.Bold)
        self.rules.append((re.compile(r"\[\+\].*$"), fmt_success))

        # 2. 步驟與進度標記 [*]
        fmt_info = QtGui.QTextCharFormat()
        fmt_info.setForeground(QtGui.QColor(theme["info"]))
        fmt_info.setFontWeight(QtGui.QFont.Weight.Bold)
        self.rules.append((re.compile(r"\[\*(?:[^\n\]]*)\](?:\s*.*)?$"), fmt_info))

        # 3. 錯誤與例外標記 [-] 或 Error / Traceback
        fmt_error = QtGui.QTextCharFormat()
        fmt_error.setForeground(QtGui.QColor(theme["error"]))
        fmt_error.setFontWeight(QtGui.QFont.Weight.Bold)
        self.rules.append((re.compile(r"\[-\].*$"), fmt_error))
        self.rules.append((re.compile(r"(?i)\b(error|exception|traceback|failed|fatal)\b"), fmt_error))

        # 4. 警示與慶祝完成標記 [!] 或 [🎉]
        fmt_warn = QtGui.QTextCharFormat()
        fmt_warn.setForeground(QtGui.QColor(theme["warning"]))
        fmt_warn.setFontWeight(QtGui.QFont.Weight.Bold)
        self.rules.append((re.compile(r"\[🎉\].*$"), fmt_warn))
        self.rules.append((re.compile(r"\[!\].*$"), fmt_warn))

        # 5. 分隔線 (===, ---, ###)
        fmt_dim = QtGui.QTextCharFormat()
        fmt_dim.setForeground(QtGui.QColor(theme["dim"]))
        self.rules.append((re.compile(r"^[\s=\-#*]{4,}$"), fmt_dim))

    def highlightBlock(self, text: str):
        for pattern, fmt in self.rules:
            for match in pattern.finditer(text):
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, fmt)


class FileManagerWidget(QWidget):
    """左側：檔案清單與 Import 管理元件"""
    files_updated = Signal()
    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.files_data: List[Dict] = []  # list of dict: path, name, size
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # 頂部工具列
        btn_layout = QHBoxLayout()
        self.btn_import_dir = QPushButton("📁 匯入目錄")
        self.btn_import_files = QPushButton("➕ 新增圖檔")
        self.btn_clear = QPushButton("🗑️ 清空")
        btn_layout.addWidget(self.btn_import_dir)
        btn_layout.addWidget(self.btn_import_files)
        btn_layout.addWidget(self.btn_clear)
        layout.addLayout(btn_layout)

        # 快速過濾與統計
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("🔍 過濾檔名 (如 Cover, Page)...")
        self.filter_edit.textChanged.connect(self.filter_table)
        layout.addWidget(self.filter_edit)

        # 檔案表格
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["檔名", "大小", "📑 TOC", "🚫 略過OCR", "🩹 Patch"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(1, 58)
        self.table.setColumnWidth(2, 62)
        self.table.setColumnWidth(3, 85)
        self.table.setColumnWidth(4, 68)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        self.table.itemChanged.connect(self.on_item_changed)
        self.table.installEventFilter(self)
        layout.addWidget(self.table)

        # 批次勾選工具
        batch_layout = QHBoxLayout()
        self.btn_auto_toc = QPushButton("標記 Cover 為 TOC")
        self.btn_clear_tags = QPushButton("清除所有標記")
        batch_layout.addWidget(self.btn_auto_toc)
        batch_layout.addWidget(self.btn_clear_tags)
        layout.addLayout(batch_layout)

        # 統計標籤
        self.lbl_stats = QLabel("圖檔數: 0 | TOC: 0 | 略過: 0 | Patch: 0")
        self.lbl_stats.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.lbl_stats)

        # 訊號綁定
        self.btn_import_dir.clicked.connect(self.import_directory)
        self.btn_import_files.clicked.connect(self.import_files)
        self.btn_clear.clicked.connect(self.clear_files)
        self.btn_auto_toc.clicked.connect(self.mark_covers_as_toc)
        self.btn_clear_tags.clicked.connect(self.clear_all_tags)

    def import_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "選取圖檔目錄", str(Path.cwd()))
        if not dir_path:
            return
        p = Path(dir_path)
        found_files = []
        for root, _, files in os.walk(p):
            for f in files:
                fp = Path(root) / f
                if fp.suffix.lower() in VALID_EXTENSIONS:
                    found_files.append(fp)
        self.add_paths(found_files)

    def import_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "選取圖檔", str(Path.cwd()),
            "Image Files (*.jpg *.jpeg *.png *.webp *.tif *.tiff *.bmp)"
        )
        if file_paths:
            self.add_paths([Path(p) for p in file_paths])

    def add_paths(self, paths: List[Path]):
        existing_paths = {d["path"] for d in self.files_data}
        new_paths = [p for p in paths if p not in existing_paths]
        for p in new_paths:
            try:
                sz = p.stat().st_size
                sz_str = f"{sz / 1024:.1f} KB" if sz < 1024 * 1024 else f"{sz / (1024 * 1024):.2f} MB"
            except Exception:
                sz_str = "0 B"
            self.files_data.append({
                "path": p,
                "name": p.name,
                "size": sz_str,
                "toc": "cover" in p.name.lower(),
                "skip": False,
                "patch": False
            })

        self.files_data.sort(key=lambda x: natural_sort_key(x["name"]))
        self.refresh_table()

    def clear_files(self):
        self.files_data.clear()
        self.refresh_table()

    def refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        filter_text = self.filter_edit.text().strip().lower()

        for d in self.files_data:
            if filter_text and filter_text not in d["name"].lower():
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)

            # 檔名
            name_item = QTableWidgetItem(d["name"])
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            name_item.setData(Qt.UserRole, str(d["path"]))
            self.table.setItem(row, 0, name_item)

            # 大小
            size_item = QTableWidgetItem(d["size"])
            size_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            size_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 1, size_item)

            # Checkboxes: TOC, Skip, Patch
            for col, key in [(2, "toc"), (3, "skip"), (4, "patch")]:
                chk_item = QTableWidgetItem()
                chk_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                chk_item.setCheckState(Qt.Checked if d[key] else Qt.Unchecked)
                self.table.setItem(row, col, chk_item)

        self.table.blockSignals(False)
        self.update_stats()
        self.files_updated.emit()

    def filter_table(self):
        self.refresh_table()

    def on_selection_changed(self):
        self.selection_changed.emit()

    def get_selected_row_indices(self) -> List[int]:
        selected_model_rows = self.table.selectionModel().selectedRows()
        return sorted([r.row() for r in selected_model_rows])

    def get_selected_row_paths(self) -> List[str]:
        paths = []
        for row in self.get_selected_row_indices():
            item = self.table.item(row, 0)
            if item:
                p = item.data(Qt.UserRole)
                if p:
                    paths.append(str(p))
        return paths

    def get_selected_row_names(self) -> List[str]:
        names = []
        for row in self.get_selected_row_indices():
            item = self.table.item(row, 0)
            if item:
                names.append(item.text())
        return names

    def delete_selected_files(self):
        selected_paths = set(self.get_selected_row_paths())
        if not selected_paths:
            return
        self.files_data = [d for d in self.files_data if str(d["path"]) not in selected_paths]
        self.refresh_table()
        self.files_updated.emit()
        self.selection_changed.emit()

    def eventFilter(self, obj, event):
        if obj == self.table and event.type() == QtCore.QEvent.KeyPress:
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                self.delete_selected_files()
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected_files()
        else:
            super().keyPressEvent(event)

    def is_path_toc(self, path: Union[str, Path]) -> bool:
        path_str = str(path)
        for d in self.files_data:
            if str(d["path"]) == path_str:
                return bool(d.get("toc"))
        return False

    def set_tags_for_selected(self, toc: Optional[bool] = None, skip: Optional[bool] = None, patch: Optional[bool] = None):
        selected_paths = set(self.get_selected_row_paths())
        if not selected_paths:
            return
        for d in self.files_data:
            if str(d["path"]) in selected_paths:
                if toc is not None:
                    d["toc"] = toc
                if skip is not None:
                    d["skip"] = skip
                if patch is not None:
                    d["patch"] = patch
        self.refresh_table()
        self.files_updated.emit()

    def open_selected_dir(self):
        paths = self.get_selected_row_paths()
        if paths:
            folder = Path(paths[0]).parent
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder.resolve())))

    def show_context_menu(self, pos: QtCore.QPoint):
        item_at_pos = self.table.itemAt(pos)
        if item_at_pos and not item_at_pos.isSelected():
            self.table.clearSelection()
            self.table.selectRow(item_at_pos.row())

        selected_paths = self.get_selected_row_paths()
        menu = QtWidgets.QMenu(self)

        if selected_paths:
            count = len(selected_paths)
            act_delete = menu.addAction(f"🗑️ 從清單移除選取的 {count} 個檔案 (Delete)")
            menu.addSeparator()
            act_mark_toc = menu.addAction("📑 標記選取為目錄頁 (TOC)")
            act_mark_skip = menu.addAction("🚫 標記選取為略過 OCR")
            act_mark_patch = menu.addAction("🩹 標記選取為 Patch 修復頁")
            menu.addSeparator()
            act_clear_tags = menu.addAction("🔄 清除選取的標記")
            menu.addSeparator()
            act_open_dir = menu.addAction("📂 開啟所在資料夾")

            action = menu.exec(self.table.viewport().mapToGlobal(pos))
            if not action:
                return

            if action == act_delete:
                self.delete_selected_files()
            elif action == act_mark_toc:
                self.set_tags_for_selected(toc=True)
            elif action == act_mark_skip:
                self.set_tags_for_selected(skip=True)
            elif action == act_mark_patch:
                self.set_tags_for_selected(patch=True)
            elif action == act_clear_tags:
                self.set_tags_for_selected(toc=False, skip=False, patch=False)
            elif action == act_open_dir:
                self.open_selected_dir()
        else:
            act_import_dir = menu.addAction("📁 匯入目錄...")
            act_import_files = menu.addAction("➕ 新增圖檔...")
            action = menu.exec(self.table.viewport().mapToGlobal(pos))
            if action == act_import_dir:
                self.import_directory()
            elif action == act_import_files:
                self.import_files()

    def on_item_changed(self, item: QTableWidgetItem):
        col = item.column()
        if col in (2, 3, 4):
            row = item.row()
            name_item = self.table.item(row, 0)
            if not name_item:
                return
            path_str = name_item.data(Qt.UserRole)
            key = {2: "toc", 3: "skip", 4: "patch"}[col]
            is_checked = (item.checkState() == Qt.Checked)
            for d in self.files_data:
                if str(d["path"]) == path_str:
                    d[key] = is_checked
                    break
            self.update_stats()
            self.files_updated.emit()

    def mark_covers_as_toc(self):
        for d in self.files_data:
            if "cover" in d["name"].lower():
                d["toc"] = True
        self.refresh_table()

    def clear_all_tags(self):
        for d in self.files_data:
            d["toc"] = False
            d["skip"] = False
            d["patch"] = False
        self.refresh_table()

    def update_stats(self):
        total = len(self.files_data)
        t_cnt = sum(1 for d in self.files_data if d["toc"])
        s_cnt = sum(1 for d in self.files_data if d["skip"])
        p_cnt = sum(1 for d in self.files_data if d["patch"])
        self.lbl_stats.setText(f"圖檔數: {total} | TOC: {t_cnt} | 略過: {s_cnt} | Patch: {p_cnt}")

    def get_first_dir(self) -> str:
        if self.files_data:
            return str(self.files_data[0]["path"].parent)
        return ""

    def get_selected_toc_paths(self) -> List[str]:
        return [str(d["path"]) for d in self.files_data if d["toc"]]

    def get_selected_skip_names(self) -> List[str]:
        return [d["name"] for d in self.files_data if d["skip"]]

    def get_selected_patch_names(self) -> List[str]:
        return [d["name"] for d in self.files_data if d["patch"]]


def get_ui_font_family_str() -> str:
    """
    獲取跨平台與繁體中文/Emoji 字型備選鏈 CSS 字串，確保在 Linux Qt6/QSS 下
    分頁與按鈕的 Emoji (🚀, 📐, 🤖 等) 及中文字型正常渲染，不出現缺字方框框。
    """
    candidate_families = [
        "Noto Sans CJK TC",
        "Noto Sans TC",
        "Microsoft JhengHei",
        "PingFang TC",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
        "Sans Serif",
        "Noto Color Emoji",
        "Segoe UI Emoji",
        "Apple Color Emoji",
    ]
    available = set(QtGui.QFontDatabase.families())
    matched = [f"'{f}'" for f in candidate_families if f in available or f == "Sans Serif"]
    if "'Noto Color Emoji'" not in matched and "Noto Color Emoji" in available:
        matched.append("'Noto Color Emoji'")
    if not matched:
        matched = ["sans-serif"]
    return ", ".join(matched)


def setup_app_fonts(app: Optional[QApplication] = None):
    """
    設定跨平台與繁體中文/Emoji 字型備選鏈 (Fallback Chain)，
    確保 Linux/PyQt6 下分頁標籤與按鈕 Emoji (🚀, 📐, 🤖, 📁 等) 及中文字元不出現方框框 (Tofu)。
    """
    if app is None:
        app = QApplication.instance()
    if not app:
        return
    font = app.font()
    available = set(QtGui.QFontDatabase.families())
    candidate_families = [
        "Noto Sans CJK TC",
        "Noto Sans TC",
        "Microsoft JhengHei",
        "PingFang TC",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
        font.family(),
        "Noto Color Emoji",
    ]
    chain = [f for f in candidate_families if f in available or f == font.family()]
    if "Noto Color Emoji" in available and "Noto Color Emoji" not in chain:
        chain.append("Noto Color Emoji")
    if chain:
        font.setFamilies(chain)
        app.setFont(font)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        setup_app_fonts()
        app_inst = QApplication.instance()
        if app_inst:
            self.setFont(app_inst.font())
        self.setWindowTitle("ScanORC - 掃描書本轉語意結構化 HTML/EPUB 控制台")
        self.resize(1060, 820)

        # 終端機色彩與字體狀態
        self.current_theme_key = "dark"
        self.current_font_size = 14
        self.available_font_sizes = [11, 12, 13, 14, 15, 16, 18, 20, 22, 24]

        # QProcess 實例
        self.process = QProcess(self)
        mode = getattr(getattr(QProcess, 'ProcessChannelMode', None), 'MergedChannels', None)
        if mode is not None:
            self.process.setProcessChannelMode(mode)
        elif hasattr(QProcess, 'MergedChannels'):
            self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.on_process_output)
        self.process.readyReadStandardError.connect(self.on_process_error_output)
        self.process.errorOccurred.connect(self.on_process_error)
        self.process.finished.connect(self.on_process_finished)

        self.setup_ui()
        self.apply_styles()

        # 預設自動偵測載入 samples 目錄
        if Path("samples").exists():
            self.file_manager.add_paths(list(Path("samples").glob("*.jpg")))

    def setup_ui(self):
        self.setup_menu_bar()
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # 主垂直分割器：上方為 (檔案列表 + 步驟面板)，下方為 (指令預覽 + 終端日誌)
        self.v_splitter = QSplitter(Qt.Vertical)

        # 上方水平分割器：左側為檔案清單，右側為步驟頁籤
        self.h_splitter = QSplitter(Qt.Horizontal)

        # 1. 左側檔案管理
        self.file_manager = FileManagerWidget()
        self.file_manager.setMinimumWidth(410)
        self.file_manager.files_updated.connect(self.on_files_updated)
        self.file_manager.selection_changed.connect(self.on_file_selection_changed)
        self.h_splitter.addWidget(self.file_manager)

        # 2. 右側步驟頁籤
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(500)
        self.h_splitter.addWidget(self.tabs)
        self.h_splitter.setStretchFactor(0, 2)
        self.h_splitter.setStretchFactor(1, 3)
        self.h_splitter.setSizes([440, 620])

        self.setup_tabs()
        self.tabs.currentChanged.connect(self.update_command_preview)

        self.v_splitter.addWidget(self.h_splitter)

        # 3. 下方終端日誌與控制台
        console_widget = QWidget()
        c_layout = QVBoxLayout(console_widget)
        c_layout.setContentsMargins(4, 4, 4, 4)
        c_layout.setSpacing(4)

        # CLI 指令預覽列
        cmd_layout = QHBoxLayout()
        cmd_layout.addWidget(QLabel("<b>即時 CLI 指令：</b>"))
        self.cmd_preview_edit = QLineEdit()
        self.cmd_preview_edit.setReadOnly(True)
        self.cmd_preview_edit.setStyleSheet("font-family: monospace; background: #f0f0f0; color: #222;")
        cmd_layout.addWidget(self.cmd_preview_edit)

        self.btn_copy_cmd = QPushButton("📋 複製指令")
        self.btn_copy_cmd.clicked.connect(self.copy_command_to_clipboard)
        cmd_layout.addWidget(self.btn_copy_cmd)
        c_layout.addLayout(cmd_layout)

        # 終端輸出視窗
        self.log_text = TerminalTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.zoom_requested.connect(self.on_terminal_zoom)
        self.highlighter = TerminalLogHighlighter(self.log_text.document(), self.current_theme_key)
        self.apply_terminal_style()
        c_layout.addWidget(self.log_text)

        # 底部狀態與操作列
        bottom_bar = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        bottom_bar.addWidget(self.progress_bar)

        self.lbl_status = QLabel("就緒")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #333;")
        bottom_bar.addWidget(self.lbl_status)
        bottom_bar.addStretch()

        # 終端機色彩與字級控制項
        bottom_bar.addWidget(QLabel("🎨 主題:"))
        self.combo_theme = QComboBox()
        self.combo_theme.setToolTip("切換終端機配色主題風格")
        for key, data in TERMINAL_THEMES.items():
            self.combo_theme.addItem(data["name"], userData=key)
        self.combo_theme.setCurrentIndex(list(TERMINAL_THEMES.keys()).index(self.current_theme_key))
        self.combo_theme.currentIndexChanged.connect(self.on_terminal_theme_changed)
        bottom_bar.addWidget(self.combo_theme)

        bottom_bar.addWidget(QLabel("🔤 字級:"))
        self.combo_font_size = QComboBox()
        self.combo_font_size.setToolTip("選擇終端機字體大小")
        for sz in self.available_font_sizes:
            self.combo_font_size.addItem(f"{sz} px", userData=sz)
        self.combo_font_size.setCurrentIndex(self.available_font_sizes.index(self.current_font_size))
        self.combo_font_size.currentIndexChanged.connect(self.on_terminal_font_size_changed)
        bottom_bar.addWidget(self.combo_font_size)

        self.btn_font_dec = QPushButton("A-")
        self.btn_font_dec.setToolTip("縮小終端機字體 (或按 Ctrl + 滾輪往下)")
        self.btn_font_dec.setFixedWidth(32)
        self.btn_font_dec.setStyleSheet("font-weight: bold; padding: 2px 4px;")
        self.btn_font_dec.clicked.connect(lambda: self.on_terminal_zoom(-1))
        bottom_bar.addWidget(self.btn_font_dec)

        self.btn_font_inc = QPushButton("A+")
        self.btn_font_inc.setToolTip("放大終端機字體 (或按 Ctrl + 滾輪往上)")
        self.btn_font_inc.setFixedWidth(32)
        self.btn_font_inc.setStyleSheet("font-weight: bold; padding: 2px 4px;")
        self.btn_font_inc.clicked.connect(lambda: self.on_terminal_zoom(1))
        bottom_bar.addWidget(self.btn_font_inc)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        bottom_bar.addWidget(sep)

        self.btn_run = QPushButton("▶ 執行當前步驟")
        self.btn_run.setStyleSheet("background-color: #007acc; color: white; font-weight: bold; padding: 6px 16px;")
        self.btn_run.clicked.connect(self.run_current_tab_step)
        bottom_bar.addWidget(self.btn_run)

        self.btn_stop = QPushButton("⏹ 中斷執行")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #cc3300; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_stop.clicked.connect(self.stop_current_process)
        bottom_bar.addWidget(self.btn_stop)

        self.btn_clear_log = QPushButton("🧹 清除日誌")
        self.btn_clear_log.clicked.connect(self.log_text.clear)
        bottom_bar.addWidget(self.btn_clear_log)

        c_layout.addLayout(bottom_bar)
        self.v_splitter.addWidget(console_widget)
        self.v_splitter.setStretchFactor(0, 3)
        self.v_splitter.setStretchFactor(1, 2)

        main_layout.addWidget(self.v_splitter)

    def setup_tabs(self):
        """設定各步驟之參數面板"""
        self.tab_oneclick = QWidget()
        self.tab_step1 = QWidget()
        self.tab_step2 = QWidget()
        self.tab_step3 = QWidget()
        self.tab_step4 = QWidget()
        self.tab_step5 = QWidget()
        self.tab_batch = QWidget()

        self.tabs.addTab(self.tab_oneclick, "🚀 一鍵端到端總控")
        self.tabs.addTab(self.tab_step1, "📐 步驟 1: 幾何轉正裁切")
        self.tabs.addTab(self.tab_step2, "📑 步驟 2: 目錄抽取 (TOC)")
        self.tabs.addTab(self.tab_step3, "🤖 步驟 3: AI 版面與 EPUB")
        self.tabs.addTab(self.tab_step4, "📁 步驟 4: 章節目錄歸檔")
        self.tabs.addTab(self.tab_step5, "🩹 步驟 5: 異常頁微創修復")
        self.tabs.addTab(self.tab_batch, "📚 批次多書處理")

        self.init_tab_oneclick()
        self.init_tab_step1()
        self.init_tab_step2()
        self.init_tab_step3()
        self.init_tab_step4()
        self.init_tab_step5()
        self.init_tab_batch()

    # -------------------------------------------------------------
    # Tab 0: 一鍵端到端總控 (process_book.py)
    # -------------------------------------------------------------
    def init_tab_oneclick(self):
        layout = QVBoxLayout(self.tab_oneclick)
        layout.addWidget(self.create_pref_toolbar(0))
        form = QFormLayout()

        self.oc_input = QLineEdit("samples")
        self.oc_input.textChanged.connect(self.update_command_preview)
        btn_browse_in = QPushButton("瀏覽...")
        btn_browse_in.clicked.connect(lambda: self.browse_dir(self.oc_input))
        h_in = QHBoxLayout()
        h_in.addWidget(self.oc_input)
        h_in.addWidget(btn_browse_in)
        form.addRow("原始圖檔目錄 (--input):", h_in)

        self.oc_output = QLineEdit("output")
        self.oc_output.textChanged.connect(self.update_command_preview)
        btn_browse_out = QPushButton("瀏覽...")
        btn_browse_out.clicked.connect(lambda: self.browse_dir(self.oc_output))
        h_out = QHBoxLayout()
        h_out.addWidget(self.oc_output)
        h_out.addWidget(btn_browse_out)
        form.addRow("成果輸出目錄 (--output):", h_out)

        self.oc_title = QLineEdit("")
        self.oc_title.setPlaceholderText("留空則自動讀取目錄或資料夾名稱")
        self.oc_title.textChanged.connect(self.update_command_preview)
        form.addRow("書籍標題 (--title):", self.oc_title)

        # 封面圖檔設定
        self.oc_cover = QLineEdit("")
        self.oc_cover.setPlaceholderText("留空則預設自動使用 Cover-01.jpg (亦相容 cover.*)")
        self.oc_cover.textChanged.connect(self.update_command_preview)
        btn_cov_browse = QPushButton("瀏覽圖檔...")
        btn_cov_browse.clicked.connect(lambda: self.browse_image_file(self.oc_cover))
        btn_cov_detect = QPushButton("🔄 自動偵測")
        btn_cov_detect.clicked.connect(lambda: self.auto_detect_cover_for_tab(self.oc_input, self.oc_cover))
        h_cov = QHBoxLayout()
        h_cov.addWidget(self.oc_cover)
        h_cov.addWidget(btn_cov_browse)
        h_cov.addWidget(btn_cov_detect)
        form.addRow("EPUB 封面圖檔 (--cover):", h_cov)

        self.oc_models_dir = QLineEdit("")
        self.oc_models_dir.setPlaceholderText("留空使用預設 (/opt/models 或 models/ 或 SCANORC_MODELS_DIR)")
        self.oc_models_dir.textChanged.connect(self.update_command_preview)
        btn_oc_models = QPushButton("瀏覽目錄...")
        btn_oc_models.clicked.connect(lambda: self.browse_dir(self.oc_models_dir))
        h_oc_mod = QHBoxLayout()
        h_oc_mod.addWidget(self.oc_models_dir)
        h_oc_mod.addWidget(btn_oc_models)
        form.addRow("模型權重目錄 (--models-dir):", h_oc_mod)

        h_opts = QHBoxLayout()
        self.oc_workers = QSpinBox()
        self.oc_workers.setRange(1, 16)
        self.oc_workers.setValue(1)
        self.oc_workers.valueChanged.connect(self.update_command_preview)
        h_opts.addWidget(QLabel("並行 Workers:"))
        h_opts.addWidget(self.oc_workers)

        self.oc_batch = QSpinBox()
        self.oc_batch.setRange(1, 8)
        self.oc_batch.setValue(2)
        self.oc_batch.valueChanged.connect(self.update_command_preview)
        h_opts.addWidget(QLabel("單頁 Batch:"))
        h_opts.addWidget(self.oc_batch)

        self.oc_dpi = QLineEdit("")
        self.oc_dpi.setPlaceholderText("留空表示 1:1 原生尺寸")
        self.oc_dpi.textChanged.connect(self.update_command_preview)
        h_opts.addWidget(QLabel("DPI:"))
        h_opts.addWidget(self.oc_dpi)

        self.oc_unsharp = QLineEdit("1x0.55+1.5+0.002")
        self.oc_unsharp.setPlaceholderText("例如 1x0.55+1.5+0.002 或 none")
        self.oc_unsharp.textChanged.connect(self.update_command_preview)
        h_opts.addWidget(QLabel("銳化:"))
        h_opts.addWidget(self.oc_unsharp)
        form.addRow("效能與尺寸參數:", h_opts)

        # images 插圖重新壓縮與尺寸縮放設定
        self.oc_compress_images = QCheckBox("啟用 images 插圖壓縮與縮放 (--compress-images)")
        self.oc_compress_images.setChecked(True)
        self.oc_compress_images.stateChanged.connect(self.update_command_preview)

        self.oc_img_quality = QSpinBox()
        self.oc_img_quality.setRange(1, 100)
        self.oc_img_quality.setValue(75)
        self.oc_img_quality.setSuffix(" %")
        self.oc_img_quality.valueChanged.connect(self.update_command_preview)

        self.oc_img_scale = QDoubleSpinBox()
        self.oc_img_scale.setRange(0.1, 2.0)
        self.oc_img_scale.setSingleStep(0.05)
        self.oc_img_scale.setValue(0.8)
        self.oc_img_scale.valueChanged.connect(self.update_command_preview)

        self.oc_img_max_dim = QSpinBox()
        self.oc_img_max_dim.setRange(0, 4096)
        self.oc_img_max_dim.setValue(0)
        self.oc_img_max_dim.setSpecialValueText("不限制 (原尺寸)")
        self.oc_img_max_dim.valueChanged.connect(self.update_command_preview)

        self.oc_img_unsharp = QLineEdit("")
        self.oc_img_unsharp.setPlaceholderText("留空關閉 (例如 1x0.55+1.5+0.002)")
        self.oc_img_unsharp.textChanged.connect(self.update_command_preview)

        h_img_opt = QHBoxLayout()
        h_img_opt.addWidget(QLabel("品質:"))
        h_img_opt.addWidget(self.oc_img_quality)
        h_img_opt.addWidget(QLabel("縮放:"))
        h_img_opt.addWidget(self.oc_img_scale)
        h_img_opt.addWidget(QLabel("最大長寬:"))
        h_img_opt.addWidget(self.oc_img_max_dim)
        h_img_opt.addWidget(QLabel("銳化:"))
        h_img_opt.addWidget(self.oc_img_unsharp)

        form.addRow("images 圖片優化:", self.oc_compress_images)
        form.addRow("圖片品質與尺寸:", h_img_opt)

        self.oc_skip = QLineEdit("")
        self.oc_skip.setPlaceholderText("自動同步檔案清單之『略過OCR』勾選項，亦可在此手動輸入")
        self.oc_skip.textChanged.connect(self.update_command_preview)
        form.addRow("略過 OCR 頁面 (--skip-ocr):", self.oc_skip)

        grid_chk = QGridLayout()
        self.oc_epub = QCheckBox("自動編譯 MathML EPUB3 (--epub)")
        self.oc_epub.setChecked(True)
        self.oc_epub.stateChanged.connect(self.update_command_preview)
        grid_chk.addWidget(self.oc_epub, 0, 0)

        self.oc_organize = QCheckBox("自動執行章節目錄歸檔 (--organize-pages)")
        self.oc_organize.setChecked(True)
        self.oc_organize.stateChanged.connect(self.update_command_preview)
        grid_chk.addWidget(self.oc_organize, 0, 1)

        self.oc_copy = QCheckBox("章節歸檔使用複製而非移動 (--copy)")
        self.oc_copy.stateChanged.connect(self.update_command_preview)
        grid_chk.addWidget(self.oc_copy, 1, 0, 1, 2)
        form.addRow("功能開關:", grid_chk)

        layout.addLayout(form)
        desc = QLabel(
            "<b>說明：</b> 一鍵自動串聯【轉正裁切 → 目錄抽取 → AI OCR & EPUB → 章節歸檔】全流程。"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #444; margin-top: 10px;")
        layout.addWidget(desc)
        layout.addStretch()

    # -------------------------------------------------------------
    # Tab 1: 幾何預處理 (preprocess_scantailor.py)
    # -------------------------------------------------------------
    def init_tab_step1(self):
        layout = QVBoxLayout(self.tab_step1)
        layout.addWidget(self.create_pref_toolbar(1))
        form = QFormLayout()

        self.s1_input = QLineEdit("samples")
        self.s1_input.setPlaceholderText("可為圖檔目錄，或檔案清單中選取的單張/多張圖檔路徑")
        self.s1_input.textChanged.connect(self.update_command_preview)
        btn_in = QPushButton("瀏覽目錄...")
        btn_in.clicked.connect(lambda: self.browse_dir(self.s1_input))
        btn_sync_s1 = QPushButton("🔄 同步清單所選")
        btn_sync_s1.setToolTip("自動填入左側檔案清單目前選取的圖檔路徑 (若未選取則填入整個目錄)")
        btn_sync_s1.clicked.connect(self.sync_step1_input)
        h_in = QHBoxLayout()
        h_in.addWidget(self.s1_input)
        h_in.addWidget(btn_in)
        h_in.addWidget(btn_sync_s1)
        form.addRow("輸入來源 (--input):", h_in)

        self.s1_output = QLineEdit("preprocessed_samples")
        self.s1_output.textChanged.connect(self.update_command_preview)
        btn_out = QPushButton("瀏覽...")
        btn_out.clicked.connect(lambda: self.browse_dir(self.s1_output))
        h_out = QHBoxLayout()
        h_out.addWidget(self.s1_output)
        h_out.addWidget(btn_out)
        form.addRow("輸出目錄 (--output):", h_out)

        h_params = QHBoxLayout()
        self.s1_color = QComboBox()
        self.s1_color.addItems(["color_grayscale", "mixed", "black_and_white"])
        self.s1_color.currentTextChanged.connect(self.update_command_preview)
        h_params.addWidget(QLabel("色彩模式:"))
        h_params.addWidget(self.s1_color)

        self.s1_despeckle = QComboBox()
        self.s1_despeckle.addItems(["normal", "cautious", "aggressive", "off"])
        self.s1_despeckle.currentTextChanged.connect(self.update_command_preview)
        h_params.addWidget(QLabel("去雜點:"))
        h_params.addWidget(self.s1_despeckle)

        self.s1_margins = QSpinBox()
        self.s1_margins.setRange(0, 100)
        self.s1_margins.setValue(1)
        self.s1_margins.valueChanged.connect(self.update_command_preview)
        h_params.addWidget(QLabel("邊距 (mm):"))
        h_params.addWidget(self.s1_margins)
        form.addRow("版面設定:", h_params)

        h_fmt = QHBoxLayout()
        self.s1_format = QComboBox()
        self.s1_format.addItems(["jpg", "png", "tif"])
        self.s1_format.currentTextChanged.connect(self.update_command_preview)
        h_fmt.addWidget(QLabel("格式:"))
        h_fmt.addWidget(self.s1_format)

        self.s1_quality = QSpinBox()
        self.s1_quality.setRange(1, 100)
        self.s1_quality.setValue(60)
        self.s1_quality.valueChanged.connect(self.update_command_preview)
        h_fmt.addWidget(QLabel("品質:"))
        h_fmt.addWidget(self.s1_quality)

        self.s1_scale = QDoubleSpinBox()
        self.s1_scale.setRange(0.1, 2.0)
        self.s1_scale.setSingleStep(0.05)
        self.s1_scale.setValue(0.8)
        self.s1_scale.valueChanged.connect(self.update_command_preview)
        h_fmt.addWidget(QLabel("縮放:"))
        h_fmt.addWidget(self.s1_scale)

        self.s1_dpi = QLineEdit("")
        self.s1_dpi.setPlaceholderText("留空為 1:1 原生尺寸")
        self.s1_dpi.textChanged.connect(self.update_command_preview)
        h_fmt.addWidget(QLabel("DPI:"))
        h_fmt.addWidget(self.s1_dpi)

        self.s1_unsharp = QLineEdit("1x0.55+1.5+0.002")
        self.s1_unsharp.setPlaceholderText("例如 1x0.55+1.5+0.002 或 none")
        self.s1_unsharp.textChanged.connect(self.update_command_preview)
        h_fmt.addWidget(QLabel("銳化:"))
        h_fmt.addWidget(self.s1_unsharp)
        form.addRow("輸出品質與尺寸:", h_fmt)

        self.s1_skip = QLineEdit("")
        self.s1_skip.setPlaceholderText("指定略過頁碼/檔名 (自動同步左側『略過OCR』勾選項，繞過 ScanTailor 直接導出原圖)")
        self.s1_skip.textChanged.connect(self.update_command_preview)
        btn_sync_s1_skip = QPushButton("🔄 同步略過清單")
        btn_sync_s1_skip.setToolTip("自動同步左側檔案清單中已勾選為『略過OCR』的檔案")
        btn_sync_s1_skip.clicked.connect(self.sync_step1_skip)
        h_skip1 = QHBoxLayout()
        h_skip1.addWidget(self.s1_skip)
        h_skip1.addWidget(btn_sync_s1_skip)
        form.addRow("略過處理頁面 (--skip-orc):", h_skip1)

        layout.addLayout(form)

        h_act = QHBoxLayout()
        btn_open_out = QPushButton("📂 開啟預處理輸出目錄")
        btn_open_out.clicked.connect(lambda: self.open_folder(self.s1_output.text()))
        h_act.addWidget(btn_open_out)
        h_act.addStretch()
        layout.addLayout(h_act)

        desc = QLabel("<b>特色：</b> 獨立書籍版心鎖定、自動旋轉校正、清除黑邊與中縫漸層陰影、1:1 原生尺寸無損輸出。")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #444; margin-top: 10px;")
        layout.addWidget(desc)
        layout.addStretch()

    # -------------------------------------------------------------
    # Tab 2: 抽取目錄結構 (extract_toc.py)
    # -------------------------------------------------------------
    def init_tab_step2(self):
        layout = QVBoxLayout(self.tab_step2)
        layout.addWidget(self.create_pref_toolbar(2))
        form = QFormLayout()

        self.s2_input = QLineEdit("")
        self.s2_input.setPlaceholderText("留空則自動搜尋 Content-xx.jpg，亦可自訂輸入")
        self.s2_input.textChanged.connect(self.update_command_preview)
        btn_in = QPushButton("選取圖檔...")
        btn_in.clicked.connect(self.browse_toc_input_files)
        h_in = QHBoxLayout()
        h_in.addWidget(self.s2_input)
        h_in.addWidget(btn_in)
        form.addRow("目錄頁圖檔 (--input):", h_in)

        self.s2_output = QLineEdit("toc.json")
        self.s2_output.textChanged.connect(self.update_command_preview)
        form.addRow("輸出 JSON 路徑 (--output):", self.s2_output)

        self.s2_title = QLineEdit("")
        self.s2_title.setPlaceholderText("選填，記錄於 JSON metadata")
        self.s2_title.textChanged.connect(self.update_command_preview)
        form.addRow("書籍標題 (--book-title):", self.s2_title)

        self.s2_models_dir = QLineEdit("")
        self.s2_models_dir.setPlaceholderText("留空使用預設 (/opt/models 或 models/ 或 SCANORC_MODELS_DIR)")
        self.s2_models_dir.textChanged.connect(self.update_command_preview)
        btn_s2_models = QPushButton("瀏覽目錄...")
        btn_s2_models.clicked.connect(lambda: self.browse_dir(self.s2_models_dir))
        h_s2_mod = QHBoxLayout()
        h_s2_mod.addWidget(self.s2_models_dir)
        h_s2_mod.addWidget(btn_s2_models)
        form.addRow("模型權重目錄 (--models-dir):", h_s2_mod)

        layout.addLayout(form)

        # 內嵌 JSON 檢視與編輯區
        box_json = QGroupBox("抽取結果檢視與微調 (toc.json)")
        box_layout = QVBoxLayout(box_json)
        self.toc_json_edit = QTextEdit()
        self.toc_json_edit.setPlaceholderText("抽取完成後此處將自動顯示結構化 JSON，亦可在此手動微調章節名稱或頁碼後儲存。")
        box_layout.addWidget(self.toc_json_edit)

        h_j_act = QHBoxLayout()
        btn_load_json = QPushButton("🔄 讀取現有 toc.json")
        btn_load_json.clicked.connect(self.load_toc_json)
        btn_save_json = QPushButton("💾 儲存修改至 toc.json")
        btn_save_json.clicked.connect(self.save_toc_json)
        h_j_act.addWidget(btn_load_json)
        h_j_act.addWidget(btn_save_json)
        h_j_act.addStretch()
        box_layout.addLayout(h_j_act)

        layout.addWidget(box_json)

    def browse_toc_input_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "選取目錄頁圖檔", str(Path.cwd()),
            "Image Files (*.jpg *.jpeg *.png *.webp)"
        )
        if files:
            self.s2_input.setText(" ".join(files))

    def load_toc_json(self):
        p = Path(self.s2_output.text().strip() or "toc.json")
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    self.toc_json_edit.setText(f.read())
            except Exception as e:
                QMessageBox.warning(self, "錯誤", f"讀取 JSON 失敗: {e}")
        else:
            QMessageBox.information(self, "提示", f"找不到檔案: {p}")

    def save_toc_json(self):
        p = Path(self.s2_output.text().strip() or "toc.json")
        content = self.toc_json_edit.toPlainText().strip()
        if not content:
            return
        try:
            # 驗證是否合規 JSON
            json.loads(content)
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            QMessageBox.information(self, "成功", f"已成功儲存 {p}")
        except Exception as e:
            QMessageBox.critical(self, "JSON 格式錯誤", f"無法儲存，JSON 語法有誤:\n{e}")

    # -------------------------------------------------------------
    # Tab 3: AI 版面分析與 OCR + HTML/EPUB 生成 (pipeline.py)
    # -------------------------------------------------------------
    def init_tab_step3(self):
        layout = QVBoxLayout(self.tab_step3)
        layout.addWidget(self.create_pref_toolbar(3))
        form = QFormLayout()

        self.s3_input = QLineEdit("preprocessed_samples")
        self.s3_input.textChanged.connect(self.update_command_preview)
        btn_in = QPushButton("瀏覽...")
        btn_in.clicked.connect(lambda: self.browse_dir(self.s3_input))
        h_in = QHBoxLayout()
        h_in.addWidget(self.s3_input)
        h_in.addWidget(btn_in)
        form.addRow("圖檔來源目錄 (--samples):", h_in)

        self.s3_output = QLineEdit("output")
        self.s3_output.textChanged.connect(self.update_command_preview)
        btn_out = QPushButton("瀏覽...")
        btn_out.clicked.connect(lambda: self.browse_dir(self.s3_output))
        h_out = QHBoxLayout()
        h_out.addWidget(self.s3_output)
        h_out.addWidget(btn_out)
        form.addRow("成果儲存目錄 (--output):", h_out)

        self.s3_title = QLineEdit("數位化掃描書本")
        self.s3_title.textChanged.connect(self.update_command_preview)
        form.addRow("書籍標題 (--title):", self.s3_title)

        self.s3_toc = QLineEdit("toc.json")
        self.s3_toc.textChanged.connect(self.update_command_preview)
        form.addRow("目錄結構 JSON (--toc):", self.s3_toc)

        # 封面圖檔設定
        self.s3_cover = QLineEdit("")
        self.s3_cover.setPlaceholderText("留空則預設自動使用 Cover-01.jpg (亦相容 cover.*)")
        self.s3_cover.textChanged.connect(self.update_command_preview)
        btn_s3_cov_browse = QPushButton("瀏覽圖檔...")
        btn_s3_cov_browse.clicked.connect(lambda: self.browse_image_file(self.s3_cover))
        btn_s3_cov_detect = QPushButton("🔄 自動偵測")
        btn_s3_cov_detect.clicked.connect(lambda: self.auto_detect_cover_for_tab(self.s3_input, self.s3_cover))
        h_s3_cov = QHBoxLayout()
        h_s3_cov.addWidget(self.s3_cover)
        h_s3_cov.addWidget(btn_s3_cov_browse)
        h_s3_cov.addWidget(btn_s3_cov_detect)
        form.addRow("EPUB 封面圖檔 (--cover):", h_s3_cov)

        self.s3_models_dir = QLineEdit("")
        self.s3_models_dir.setPlaceholderText("留空使用預設 (/opt/models 或 models/ 或 SCANORC_MODELS_DIR)")
        self.s3_models_dir.textChanged.connect(self.update_command_preview)
        btn_s3_models = QPushButton("瀏覽目錄...")
        btn_s3_models.clicked.connect(lambda: self.browse_dir(self.s3_models_dir))
        h_s3_mod = QHBoxLayout()
        h_s3_mod.addWidget(self.s3_models_dir)
        h_s3_mod.addWidget(btn_s3_models)
        form.addRow("模型權重目錄 (--models-dir):", h_s3_mod)

        h_perf = QHBoxLayout()
        self.s3_workers = QSpinBox()
        self.s3_workers.setRange(1, 16)
        self.s3_workers.setValue(1)
        self.s3_workers.valueChanged.connect(self.update_command_preview)
        h_perf.addWidget(QLabel("Workers 並行數:"))
        h_perf.addWidget(self.s3_workers)

        self.s3_batch = QSpinBox()
        self.s3_batch.setRange(1, 8)
        self.s3_batch.setValue(2)
        self.s3_batch.valueChanged.connect(self.update_command_preview)
        h_perf.addWidget(QLabel("單頁 Batch:"))
        h_perf.addWidget(self.s3_batch)
        form.addRow("推理並行設定:", h_perf)

        # images 插圖重新壓縮與尺寸縮放設定
        self.s3_compress_images = QCheckBox("啟用 images 插圖壓縮與縮放 (--compress-images)")
        self.s3_compress_images.setChecked(True)
        self.s3_compress_images.stateChanged.connect(self.update_command_preview)

        self.s3_img_quality = QSpinBox()
        self.s3_img_quality.setRange(1, 100)
        self.s3_img_quality.setValue(75)
        self.s3_img_quality.setSuffix(" %")
        self.s3_img_quality.valueChanged.connect(self.update_command_preview)

        self.s3_img_scale = QDoubleSpinBox()
        self.s3_img_scale.setRange(0.1, 2.0)
        self.s3_img_scale.setSingleStep(0.05)
        self.s3_img_scale.setValue(0.8)
        self.s3_img_scale.valueChanged.connect(self.update_command_preview)

        self.s3_img_max_dim = QSpinBox()
        self.s3_img_max_dim.setRange(0, 4096)
        self.s3_img_max_dim.setValue(0)
        self.s3_img_max_dim.setSpecialValueText("不限制 (原尺寸)")
        self.s3_img_max_dim.valueChanged.connect(self.update_command_preview)

        self.s3_img_unsharp = QLineEdit("")
        self.s3_img_unsharp.setPlaceholderText("留空關閉 (例如 1x0.55+1.5+0.002)")
        self.s3_img_unsharp.textChanged.connect(self.update_command_preview)

        h_s3_img_opt = QHBoxLayout()
        h_s3_img_opt.addWidget(QLabel("品質:"))
        h_s3_img_opt.addWidget(self.s3_img_quality)
        h_s3_img_opt.addWidget(QLabel("縮放:"))
        h_s3_img_opt.addWidget(self.s3_img_scale)
        h_s3_img_opt.addWidget(QLabel("最大長寬:"))
        h_s3_img_opt.addWidget(self.s3_img_max_dim)
        h_s3_img_opt.addWidget(QLabel("銳化:"))
        h_s3_img_opt.addWidget(self.s3_img_unsharp)

        form.addRow("images 圖片優化:", self.s3_compress_images)
        form.addRow("圖片品質與尺寸:", h_s3_img_opt)

        self.s3_skip = QLineEdit("")
        self.s3_skip.setPlaceholderText("自動同步檔案清單之『略過OCR』勾選項，亦可在此手動輸入")
        self.s3_skip.textChanged.connect(self.update_command_preview)
        form.addRow("略過 OCR 頁面 (--skip-ocr):", self.s3_skip)

        h_chk = QHBoxLayout()
        self.s3_epub = QCheckBox("調用 Pandoc 編譯 MathML EPUB (--epub)")
        self.s3_epub.setChecked(True)
        self.s3_epub.stateChanged.connect(self.update_command_preview)
        h_chk.addWidget(self.s3_epub)

        self.s3_cover_first = QCheckBox("封面/目錄優先排序 (--cover-first)")
        self.s3_cover_first.setChecked(True)
        self.s3_cover_first.stateChanged.connect(self.update_command_preview)
        h_chk.addWidget(self.s3_cover_first)
        form.addRow("排版選項:", h_chk)

        layout.addLayout(form)

        # 快捷開啟成果按鈕
        box_quick = QGroupBox("成果快速檢視與工具")
        grid_q = QGridLayout(box_quick)
        btn_open_html = QPushButton("🌐 在瀏覽器開啟 book.html")
        btn_open_epub = QPushButton("📖 開啟 book.epub")
        btn_rebuild_epub = QPushButton("⚡ 從 HTML 直接重編 EPUB")
        btn_rebuild_epub.setToolTip("調用 html_to_epub.py，免跑 AI/OCR 於 1~2 秒內直接將 HTML 與 images 打包為最新 EPUB")
        btn_open_out = QPushButton("📂 開啟 output 目錄")
        btn_open_html.clicked.connect(self.open_generated_html)
        btn_open_epub.clicked.connect(self.open_generated_epub)
        btn_rebuild_epub.clicked.connect(self.rebuild_epub_from_html)
        btn_open_out.clicked.connect(lambda: self.open_folder(self.s3_output.text()))
        grid_q.addWidget(btn_open_html, 0, 0)
        grid_q.addWidget(btn_open_epub, 0, 1)
        grid_q.addWidget(btn_rebuild_epub, 1, 0)
        grid_q.addWidget(btn_open_out, 1, 1)
        layout.addWidget(box_quick)

        layout.addStretch()

    # -------------------------------------------------------------
    # Tab 4: 章節目錄歸檔 (extract_toc.py --organize-pages)
    # -------------------------------------------------------------
    def init_tab_step4(self):
        layout = QVBoxLayout(self.tab_step4)
        layout.addWidget(self.create_pref_toolbar(4))
        form = QFormLayout()

        self.s4_target = QLineEdit("preprocessed_samples")
        self.s4_target.textChanged.connect(self.update_command_preview)
        btn_t = QPushButton("瀏覽...")
        btn_t.clicked.connect(lambda: self.browse_dir(self.s4_target))
        btn_sync_t = QPushButton("🔄 同步清單目錄")
        btn_sync_t.setToolTip("將待歸檔目錄同步為左側檔案清單中圖檔所在目錄")
        btn_sync_t.clicked.connect(self.sync_step4_target_dir)
        h_t = QHBoxLayout()
        h_t.addWidget(self.s4_target)
        h_t.addWidget(btn_t)
        h_t.addWidget(btn_sync_t)
        form.addRow("待歸檔圖檔目錄 (--organize-pages):", h_t)

        self.s4_input = QLineEdit("")
        self.s4_input.setPlaceholderText("留空自動搜尋 Content-xx.jpg 或指定 toc.json")
        self.s4_input.textChanged.connect(self.update_command_preview)
        btn_sync_toc = QPushButton("🔄 同步清單 TOC")
        btn_sync_toc.setToolTip("自動同步左側檔案清單中已勾選為『TOC』的目錄圖檔")
        btn_sync_toc.clicked.connect(self.sync_step4_toc_input)
        h_in4 = QHBoxLayout()
        h_in4.addWidget(self.s4_input)
        h_in4.addWidget(btn_sync_toc)
        form.addRow("目錄來源 (--input):", h_in4)

        self.s4_copy = QCheckBox("使用複製代替移動 (--copy)")
        self.s4_copy.stateChanged.connect(self.update_command_preview)
        form.addRow("搬移模式:", self.s4_copy)

        layout.addLayout(form)
        desc = QLabel(
            "<b>說明：</b> 依據目錄結構抽取結果，自動建立章節專屬子資料夾（如 <code>01-第一章</code>），並將內文圖檔精準歸入對應目錄。"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #444; margin-top: 10px;")
        layout.addWidget(desc)
        layout.addStretch()

    # -------------------------------------------------------------
    # Tab 5: 頁面微創替換與校對 (patch_pages.py)
    # -------------------------------------------------------------
    def init_tab_step5(self):
        layout = QVBoxLayout(self.tab_step5)
        layout.addWidget(self.create_pref_toolbar(5))
        form = QFormLayout()

        self.s5_html = QLineEdit("output/book.html")
        self.s5_html.textChanged.connect(self.update_command_preview)
        btn_h = QPushButton("瀏覽...")
        btn_h.clicked.connect(self.browse_patch_html)
        h_h = QHBoxLayout()
        h_h.addWidget(self.s5_html)
        h_h.addWidget(btn_h)
        form.addRow("待修復 HTML (--html):", h_h)

        self.s5_pages = QLineEdit("")
        self.s5_pages.setPlaceholderText("指定待替換頁碼或檔名，如: 48 60 或 Page-048.jpg (自動同步左側『Patch』勾選)")
        self.s5_pages.textChanged.connect(self.update_command_preview)
        form.addRow("欲替換為原圖的頁面 (--pages):", self.s5_pages)

        self.s5_source = QLineEdit("output/preprocessed preprocessed_samples samples")
        self.s5_source.textChanged.connect(self.update_command_preview)
        form.addRow("原始圖檔搜尋目錄 (--source):", self.s5_source)

        h_chk = QHBoxLayout()
        self.s5_epub = QCheckBox("替換後自動重新編譯 EPUB (--epub)")
        self.s5_epub.setChecked(True)
        self.s5_epub.stateChanged.connect(self.update_command_preview)
        h_chk.addWidget(self.s5_epub)

        self.s5_backup = QCheckBox("自動建立備份檔案 (--backup)")
        self.s5_backup.setChecked(True)
        self.s5_backup.stateChanged.connect(self.update_command_preview)
        h_chk.addWidget(self.s5_backup)
        form.addRow("選項:", h_chk)

        layout.addLayout(form)

        # 專屬微創操作按鈕組
        box_ops = QGroupBox("微創校對快捷動作")
        h_ops = QHBoxLayout(box_ops)

        btn_list_status = QPushButton("📋 檢視全書頁面狀態 (--list)")
        btn_list_status.clicked.connect(self.run_patch_list)
        btn_revert = QPushButton("⏮ 一鍵還原至備份版本 (--revert)")
        btn_revert.clicked.connect(self.run_patch_revert)

        h_ops.addWidget(btn_list_status)
        h_ops.addWidget(btn_revert)
        layout.addWidget(box_ops)

        desc = QLabel(
            "<b>說明：</b> 僅微創置換異常頁面的 <code>&lt;section&gt;</code> 為全頁原圖嵌入，"
            "完整保留目錄錨點跳轉，並在 <b>1~2 秒內</b>重新生成最新 EPUB，徹底免除重跑全書 OCR！"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #444; margin-top: 10px;")
        layout.addWidget(desc)
        layout.addStretch()

    def browse_patch_html(self):
        f, _ = QFileDialog.getOpenFileName(self, "選取 HTML 檔案", str(Path.cwd()), "HTML Files (*.html)")
        if f:
            self.s5_html.setText(f)

    def run_patch_list(self):
        html_file = self.s5_html.text().strip() or "output/book.html"
        cmd = [self.get_python_bin(), self.get_script_path("patch_pages.py"), "--html", html_file, "--list"]
        self.execute_cmd_list(cmd)

    def run_patch_revert(self):
        html_file = self.s5_html.text().strip() or "output/book.html"
        ret = QMessageBox.question(self, "確認還原", f"確定要從備份檔還原 {html_file} 嗎？", QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            cmd = [self.get_python_bin(), self.get_script_path("patch_pages.py"), "--html", html_file, "--revert"]
            if not self.s5_epub.isChecked():
                cmd.append("--no-epub")
            self.execute_cmd_list(cmd)

    # -------------------------------------------------------------
    # Tab 6: 批次多書處理 (batch_process.py)
    # -------------------------------------------------------------
    def init_tab_batch(self):
        layout = QVBoxLayout(self.tab_batch)
        layout.addWidget(self.create_pref_toolbar(6))
        form = QFormLayout()

        # 1. 書籍母目錄
        self.batch_input_dir = QLineEdit("")
        self.batch_input_dir.setPlaceholderText("指定包含多本書籍子目錄之母目錄 (例如: /data/books/)")
        self.batch_input_dir.textChanged.connect(self.update_command_preview)
        btn_b_in = QPushButton("瀏覽母目錄...")
        btn_b_in.clicked.connect(lambda: self.browse_dir(self.batch_input_dir))
        h_b_in = QHBoxLayout()
        h_b_in.addWidget(self.batch_input_dir)
        h_b_in.addWidget(btn_b_in)
        form.addRow("書籍母目錄 (--input-dir):", h_b_in)

        # 2. 自選多目錄清單
        self.batch_dirs = QLineEdit("")
        self.batch_dirs.setPlaceholderText("明確指定多個書籍目錄 (以空白分隔；若已指定母目錄則可留空)")
        self.batch_dirs.textChanged.connect(self.update_command_preview)
        btn_b_dirs = QPushButton("選取目錄...")
        btn_b_dirs.clicked.connect(self.browse_batch_dirs)
        h_b_dirs = QHBoxLayout()
        h_b_dirs.addWidget(self.batch_dirs)
        h_b_dirs.addWidget(btn_b_dirs)
        form.addRow("自選多目錄 (--dirs):", h_b_dirs)

        # 3. 掃描過濾與目錄偏好設定檔名
        h_pat_cfg = QHBoxLayout()
        self.batch_pattern = QLineEdit("*")
        self.batch_pattern.setPlaceholderText("子目錄過濾規則 (預設: *)")
        self.batch_pattern.textChanged.connect(self.update_command_preview)
        h_pat_cfg.addWidget(QLabel("過濾 Pattern:"))
        h_pat_cfg.addWidget(self.batch_pattern)

        self.batch_config_name = QLineEdit("book_config.json")
        self.batch_config_name.setPlaceholderText("各書專屬設定檔名 (預設: book_config.json)")
        self.batch_config_name.textChanged.connect(self.update_command_preview)
        h_pat_cfg.addWidget(QLabel("各書設定檔名:"))
        h_pat_cfg.addWidget(self.batch_config_name)
        form.addRow("掃描過濾與設定檔:", h_pat_cfg)

        # 4. 批次成果總輸出目錄
        self.batch_output_root = QLineEdit("output_batch")
        self.batch_output_root.textChanged.connect(self.update_command_preview)
        btn_b_out = QPushButton("瀏覽...")
        btn_b_out.clicked.connect(lambda: self.browse_dir(self.batch_output_root))
        h_b_out = QHBoxLayout()
        h_b_out.addWidget(self.batch_output_root)
        h_b_out.addWidget(btn_b_out)
        form.addRow("總成果輸出目錄 (--output-root):", h_b_out)

        # 5. 排程與容錯控制開關
        grid_batch_ops = QGridLayout()
        self.batch_skip_existing = QCheckBox("斷點續傳：已有成果則跳過 (--skip-existing)")
        self.batch_skip_existing.setChecked(True)
        self.batch_skip_existing.stateChanged.connect(self.update_command_preview)
        grid_batch_ops.addWidget(self.batch_skip_existing, 0, 0)

        self.batch_use_existing_toc = QCheckBox("沿用書籍目錄中既有 toc.json (--use-existing-toc)")
        self.batch_use_existing_toc.setChecked(True)
        self.batch_use_existing_toc.stateChanged.connect(self.update_command_preview)
        grid_batch_ops.addWidget(self.batch_use_existing_toc, 0, 1)

        self.batch_force_toc = QCheckBox("強制重新以 AI 抽取目錄 (--force-toc)")
        self.batch_force_toc.stateChanged.connect(self.update_command_preview)
        grid_batch_ops.addWidget(self.batch_force_toc, 1, 0)

        self.batch_continue_on_error = QCheckBox("單書失敗時繼續下一本 (--continue-on-error)")
        self.batch_continue_on_error.setChecked(True)
        self.batch_continue_on_error.stateChanged.connect(self.update_command_preview)
        grid_batch_ops.addWidget(self.batch_continue_on_error, 1, 1)

        self.batch_dry_run = QCheckBox("排程預覽：不實際執行運算 (--dry-run)")
        self.batch_dry_run.stateChanged.connect(self.update_command_preview)
        grid_batch_ops.addWidget(self.batch_dry_run, 2, 0)

        form.addRow("排程與容錯策略:", grid_batch_ops)

        # 6. 全書預設通用參數群組
        box_common = QGroupBox("全書預設通用參數 (書籍若無專屬設定則套用此處)")
        form_common = QFormLayout(box_common)

        h_b_perf = QHBoxLayout()
        self.batch_workers = QSpinBox()
        self.batch_workers.setRange(1, 16)
        self.batch_workers.setValue(1)
        self.batch_workers.valueChanged.connect(self.update_command_preview)
        h_b_perf.addWidget(QLabel("單書 Workers:"))
        h_b_perf.addWidget(self.batch_workers)

        self.batch_batch_size = QSpinBox()
        self.batch_batch_size.setRange(1, 8)
        self.batch_batch_size.setValue(2)
        self.batch_batch_size.valueChanged.connect(self.update_command_preview)
        h_b_perf.addWidget(QLabel("單頁 Batch:"))
        h_b_perf.addWidget(self.batch_batch_size)

        self.batch_dpi = QLineEdit("")
        self.batch_dpi.setPlaceholderText("留空表示 1:1 原生尺寸")
        self.batch_dpi.textChanged.connect(self.update_command_preview)
        h_b_perf.addWidget(QLabel("DPI:"))
        h_b_perf.addWidget(self.batch_dpi)

        self.batch_unsharp = QLineEdit("1x0.55+1.5+0.002")
        self.batch_unsharp.setPlaceholderText("例如 1x0.55+1.5+0.002 或 none")
        self.batch_unsharp.textChanged.connect(self.update_command_preview)
        h_b_perf.addWidget(QLabel("銳化:"))
        h_b_perf.addWidget(self.batch_unsharp)
        form_common.addRow("效能與尺寸:", h_b_perf)

        grid_common_chk = QGridLayout()
        self.batch_compress_images = QCheckBox("壓縮 images 插圖")
        self.batch_compress_images.setChecked(True)
        self.batch_compress_images.stateChanged.connect(self.update_command_preview)
        grid_common_chk.addWidget(self.batch_compress_images, 0, 0)

        self.batch_epub = QCheckBox("編譯 EPUB3")
        self.batch_epub.setChecked(True)
        self.batch_epub.stateChanged.connect(self.update_command_preview)
        grid_common_chk.addWidget(self.batch_epub, 0, 1)

        self.batch_organize = QCheckBox("依章節歸檔圖檔")
        self.batch_organize.setChecked(True)
        self.batch_organize.stateChanged.connect(self.update_command_preview)
        grid_common_chk.addWidget(self.batch_organize, 1, 0)

        self.batch_copy = QCheckBox("歸檔複製代替移動")
        self.batch_copy.stateChanged.connect(self.update_command_preview)
        grid_common_chk.addWidget(self.batch_copy, 1, 1)
        form_common.addRow("功能開關:", grid_common_chk)

        self.batch_models_dir = QLineEdit("")
        self.batch_models_dir.setPlaceholderText("留空使用預設 (/opt/models 或 models/ 或 SCANORC_MODELS_DIR)")
        self.batch_models_dir.textChanged.connect(self.update_command_preview)
        btn_batch_models = QPushButton("瀏覽目錄...")
        btn_batch_models.clicked.connect(lambda: self.browse_dir(self.batch_models_dir))
        h_b_mod = QHBoxLayout()
        h_b_mod.addWidget(self.batch_models_dir)
        h_b_mod.addWidget(btn_batch_models)
        form_common.addRow("模型權重目錄 (--models-dir):", h_b_mod)

        layout.addLayout(form)
        layout.addWidget(box_common)

        # 7. 批次輔助動作與成果檢視
        box_b_act = QGroupBox("批次輔助動作與成果檢視")
        h_b_act = QHBoxLayout(box_b_act)

        btn_run_dry = QPushButton("🔍 執行排程預覽 (--dry-run)")
        btn_run_dry.setToolTip("以當前設定進行 Dry-run 模擬掃描，於終端機顯示待處理書籍清單與各書設定來源")
        btn_run_dry.clicked.connect(self.run_batch_dry_run)
        h_b_act.addWidget(btn_run_dry)

        btn_view_report = QPushButton("📊 檢視批次報告 (batch_report.json)")
        btn_view_report.clicked.connect(self.view_batch_report)
        h_b_act.addWidget(btn_view_report)

        btn_open_batch_dir = QPushButton("📂 開啟成果目錄")
        btn_open_batch_dir.clicked.connect(lambda: self.open_folder(self.batch_output_root.text()))
        h_b_act.addWidget(btn_open_batch_dir)

        layout.addWidget(box_b_act)

        desc = QLabel(
            "<b>說明：</b> 依序佇列批次處理多個書籍目錄。優先讀取各書目錄內之 <code>book_config.json</code>，"
            "若無則套用全書通用預設值；自動優先沿用目錄中現成之 <code>toc.json</code>，支援斷點續傳。"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #444; margin-top: 6px;")
        layout.addWidget(desc)
        layout.addStretch()

    def browse_batch_dirs(self):
        """開啟資料夾選擇框，將使用者選取之路徑加入自選目錄清單"""
        d = QFileDialog.getExistingDirectory(self, "選取書籍目錄", str(Path.cwd()))
        if d:
            rel = format_rel_path(d)
            current = self.batch_dirs.text().strip()
            if current:
                self.batch_dirs.setText(f"{current} {rel}")
            else:
                self.batch_dirs.setText(rel)

    def run_batch_dry_run(self):
        """以目前設定執行 Dry-run 模擬排程"""
        cmd = self.get_current_cmd_list()
        if "--dry-run" not in cmd:
            cmd.append("--dry-run")
        self.execute_cmd_list(cmd)

    def view_batch_report(self):
        """檢視批次執行產出之 batch_report.json"""
        out_root = Path(self.batch_output_root.text().strip() or "output_batch")
        rpt_file = out_root / "batch_report.json"
        if not rpt_file.exists():
            QMessageBox.information(self, "提示", f"尚未產生批次報告檔案：\n{rpt_file}\n請先執行批次處理任務。")
            return
        try:
            content = rpt_file.read_text(encoding="utf-8")
            self.log_text.append(f"\n==================================================")
            self.log_text.append(f"📊 批次執行報告: {rpt_file}")
            self.log_text.append(f"==================================================")
            self.log_text.append(content)
            self.lbl_status.setText(f"已於終端機顯示批次報告: {rpt_file.name}")
        except Exception as e:
            QMessageBox.warning(self, "讀取失敗", f"無法讀取批次報告:\n{e}")

    # -------------------------------------------------------------
    # 輔助與事件處理
    # -------------------------------------------------------------
    def get_script_path(self, script_name: str) -> str:
        """取得目標腳本的正確絕對路徑，確保於非專案根目錄或透過 deb 安裝調用時不會遺失檔案"""
        cand1 = project_root / script_name
        if cand1.exists():
            return str(cand1.resolve())
        cand2 = Path("/opt/scanorc") / script_name
        if cand2.exists():
            return str(cand2.resolve())
        cand3 = Path.cwd() / script_name
        if cand3.exists():
            return str(cand3.resolve())
        return str(cand1)

    def get_python_bin(self) -> str:
        """取得適當的 Python 直譯器 (依序檢查自訂環境變數、專案/系統 venv、Anaconda/Conda 環境、原啟動環境與當前環境)"""
        candidates = []

        # 0. 使用者明確指定的環境變數 (SCANORC_PYTHON 或 SCANORC_AI_PYTHON)
        for env_key in ("SCANORC_PYTHON", "SCANORC_AI_PYTHON"):
            val = os.environ.get(env_key)
            if val and Path(val).is_file():
                candidates.append(Path(val))

        # 1. 專案或系統專屬虛擬環境
        candidates.extend([
            project_root / ".venv" / "bin" / "python",
            Path("/opt/scanorc/.venv/bin/python"),
            Path.home() / ".local/share/scanorc/venv" / "bin" / "python",
        ])

        # 2. 常用 Anaconda / Miniconda 深度學習環境
        candidates.extend([
            Path("/opt/anaconda3/bin/python"),
            Path.home() / "anaconda3/bin/python",
            Path("/opt/miniconda3/bin/python"),
            Path.home() / "miniconda3/bin/python",
        ])

        # 3. 若是透過 GUI bootstrap 切換環境，優先保留原始具有深度學習相依的 Python
        orig_py = os.environ.get("_SCANORC_ORIGINAL_PYTHON")
        if orig_py:
            candidates.append(Path(orig_py))

        # 4. 若目前已啟用虛擬環境
        if "VIRTUAL_ENV" in os.environ:
            candidates.append(Path(os.environ["VIRTUAL_ENV"]) / "bin" / "python")

        # 5. 若環境中有 CONDA_PREFIX
        if "CONDA_PREFIX" in os.environ:
            candidates.append(Path(os.environ["CONDA_PREFIX"]) / "bin" / "python")

        # 6. 當前工作目錄下的 .venv (開發備援)
        candidates.append(Path.cwd() / ".venv" / "bin" / "python")

        for c in candidates:
            if c.is_file() and os.access(c, os.X_OK):
                return str(c.resolve())

        return sys.executable

    def browse_dir(self, line_edit: QLineEdit):
        p = QFileDialog.getExistingDirectory(self, "選取目錄", str(Path.cwd()))
        if p:
            line_edit.setText(p)

    def browse_image_file(self, line_edit: QLineEdit):
        """開啟檔案選擇對話框選取圖片檔案"""
        f, _ = QFileDialog.getOpenFileName(
            self, "選取封面圖檔", str(Path.cwd()),
            "圖片檔案 (*.jpg *.jpeg *.png *.webp);;所有檔案 (*.*)"
        )
        if f:
            line_edit.setText(format_rel_path(Path(f)))

    def auto_detect_cover_for_tab(self, dir_input: QLineEdit, cover_input: QLineEdit):
        """在指定的輸入或輸出目錄中自動偵測 Cover 圖檔"""
        target_dir_str = dir_input.text().strip() or "samples"
        search_dirs = [Path(target_dir_str)]
        out_dir = Path("output")
        if out_dir.exists():
            search_dirs.append(out_dir)
            search_dirs.append(out_dir / "preprocessed")
        cur_dir = Path(".")
        search_dirs.append(cur_dir)

        cover_patterns = [
            re.compile(r"^cover\.(jpg|jpeg|png|webp)$", re.IGNORECASE),
            re.compile(r"^cover[-_]?\d+\.(jpg|jpeg|png|webp)$", re.IGNORECASE),
            re.compile(r"^封面\.(jpg|jpeg|png|webp)$", re.IGNORECASE),
            re.compile(r"^front[-_]?cover\.(jpg|jpeg|png|webp)$", re.IGNORECASE),
        ]

        found = None
        for d in search_dirs:
            if not d.exists():
                continue
            if d.is_file():
                for pat in cover_patterns:
                    if pat.match(d.name):
                        found = d
                        break
            elif d.is_dir():
                for root, _, files in os.walk(d):
                    for fn in sorted(files):
                        for pat in cover_patterns:
                            if pat.match(fn):
                                found = Path(root) / fn
                                break
                        if found:
                            break
                    if found:
                        break
            if found:
                break

        if found:
            rel = format_rel_path(found)
            cover_input.setText(rel)
            self.lbl_status.setText(f"已自動偵測並設定封面圖檔: {rel}")
        else:
            QMessageBox.information(self, "提示", f"在目錄 '{target_dir_str}' 及相關路徑中未找到符合 Cover-*.jpg 或 cover.* 的圖檔。")

    def on_file_selection_changed(self):
        """左側表格列選取變更時，依參數需求傳遞至步驟 1 與步驟 4，並智能代入封面"""
        selected_paths = self.file_manager.get_selected_row_paths()
        if not selected_paths:
            return

        # 1. 傳遞至步驟 1: --input 填入所選取的圖檔路徑清單
        self.s1_input.setText(format_paths_arg(selected_paths))

        # 2. 傳遞至步驟 4:
        # a. 待歸檔圖檔目錄 (--organize-pages): 若目前為空才填入所選檔案所在目錄
        if not self.s4_target.text().strip():
            first_parent = Path(selected_paths[0]).parent
            self.s4_target.setText(format_rel_path(first_parent))

        # b. 目錄來源 (--input): 若選取項目包含 TOC 標記或 Cover/目錄關鍵字，自動帶入
        selected_toc = [
            p for p in selected_paths
            if any(k in Path(p).name.lower() for k in ("cover", "toc", "mulu", "catalogue"))
            or self.file_manager.is_path_toc(p)
        ]
        if selected_toc:
            self.s4_input.setText(format_paths_arg(selected_toc))

        # c. 若選取單一圖檔且為 Cover/封面圖，且封面設定為空時自動代入
        if len(selected_paths) == 1:
            p_sel = Path(selected_paths[0])
            if re.search(r"(^cover|^封面|^front[-_]?cover)", p_sel.stem.lower()):
                rel_cov = format_rel_path(p_sel)
                if not self.oc_cover.text().strip():
                    self.oc_cover.setText(rel_cov)
                if not self.s3_cover.text().strip():
                    self.s3_cover.setText(rel_cov)

        self.update_command_preview()

    def sync_step1_input(self):
        """主動將左側檔案清單選取的檔案或目錄同步至步驟 1 (--input)"""
        selected_paths = self.file_manager.get_selected_row_paths()
        if selected_paths:
            self.s1_input.setText(format_paths_arg(selected_paths))
            self.lbl_status.setText(f"已同步 {len(selected_paths)} 個選取圖檔至步驟 1 輸入")
        else:
            first_dir = self.file_manager.get_first_dir()
            if first_dir:
                self.s1_input.setText(format_rel_path(first_dir))
                self.lbl_status.setText(f"已同步圖檔目錄至步驟 1 輸入: {format_rel_path(first_dir)}")
            else:
                self.s1_input.setText("samples")
                self.lbl_status.setText("步驟 1 輸入重設為預設目錄 (samples)")
        self.update_command_preview()

    def sync_step1_skip(self):
        """主動將左側檔案清單中勾選為『略過OCR』的檔案同步至步驟 1 (--skip-orc)"""
        skip_files = self.file_manager.get_selected_skip_names()
        if skip_files:
            self.s1_skip.setText(" ".join(skip_files))
            self.lbl_status.setText(f"已同步 {len(skip_files)} 個略過處理頁面至步驟 1")
        else:
            selected_names = self.file_manager.get_selected_row_names()
            if selected_names:
                self.s1_skip.setText(" ".join(selected_names))
                self.lbl_status.setText(f"步驟 1 已將選取的 {len(selected_names)} 個圖檔設為略過處理")
            else:
                self.s1_skip.setText("")
                self.lbl_status.setText("步驟 1 略過清單已清空")
        self.update_command_preview()

    def sync_step4_target_dir(self):
        """主動將左側檔案清單中圖檔所在目錄同步至步驟 4 待歸檔目錄 (--organize-pages)"""
        selected_paths = self.file_manager.get_selected_row_paths()
        if selected_paths:
            target_dir = format_rel_path(Path(selected_paths[0]).parent)
        else:
            first_dir = self.file_manager.get_first_dir()
            target_dir = format_rel_path(first_dir) if first_dir else "preprocessed_samples"
        self.s4_target.setText(target_dir)
        self.lbl_status.setText(f"已同步步驟 4 待歸檔目錄: {target_dir}")
        self.update_command_preview()

    def sync_step4_toc_input(self):
        """主動將左側清單中 TOC 標記或 Cover 圖檔同步至步驟 4 目錄來源 (--input)"""
        # 1. 優先取勾選 TOC 的圖檔
        toc_files = self.file_manager.get_selected_toc_paths()
        # 2. 若無勾選，尋找被選取列中含 cover/toc 的圖檔
        if not toc_files:
            selected_paths = self.file_manager.get_selected_row_paths()
            toc_files = [p for p in selected_paths if any(k in Path(p).name.lower() for k in ("cover", "toc", "mulu", "catalogue"))]
        # 3. 若仍無，搜尋全清單中含 cover 的圖檔
        if not toc_files:
            toc_files = [str(d["path"]) for d in self.file_manager.files_data if "cover" in d["name"].lower()]

        if toc_files:
            self.s4_input.setText(format_paths_arg(toc_files))
            self.lbl_status.setText(f"已同步 {len(toc_files)} 個 TOC 檔案至步驟 4 目錄來源")
        else:
            QMessageBox.information(self, "提示", "未找到標記為 TOC 的圖檔，請先在左側勾選『TOC』或選取目錄圖檔！")
        self.update_command_preview()

    def on_files_updated(self):
        """左側檔案清單更新時，自動同步參數預設值"""
        if not self.file_manager.files_data:
            self.s1_input.setText("samples")
            self.s1_skip.setText("")
            self.oc_input.setText("samples")
            self.s4_target.setText("preprocessed_samples")
            self.s4_input.setText("preprocessed_samples/Cover-06.jpg")
            self.s2_input.setText("")
            self.oc_skip.setText("")
            self.s3_skip.setText("")
            self.s5_pages.setText("")
            self.update_command_preview()
            return

        first_dir = self.file_manager.get_first_dir()
        if first_dir:
            formatted_first_dir = format_rel_path(first_dir)
            if not self.oc_input.text() or self.oc_input.text() == "samples":
                self.oc_input.setText(formatted_first_dir)
            if not self.s1_input.text() or self.s1_input.text() == "samples":
                self.s1_input.setText(formatted_first_dir)
            if not self.s4_target.text().strip():
                self.s4_target.setText("preprocessed_samples")

        # 同步 TOC 勾選
        toc_files = self.file_manager.get_selected_toc_paths()
        if toc_files:
            formatted_toc = format_paths_arg(toc_files)
            self.s2_input.setText(formatted_toc)
            self.s4_input.setText(formatted_toc)

        # 同步 Skip-OCR 勾選
        skip_files = self.file_manager.get_selected_skip_names()
        skip_str = " ".join(skip_files)
        self.oc_skip.setText(skip_str)
        self.s1_skip.setText(skip_str)
        self.s3_skip.setText(skip_str)

        # 同步 Patch 勾選
        patch_files = self.file_manager.get_selected_patch_names()
        patch_str = " ".join(patch_files)
        self.s5_pages.setText(patch_str)

        self.update_command_preview()

    def get_current_cmd_list(self) -> List[str]:
        """根據目前分頁與欄位產生完整命令清單"""
        idx = self.tabs.currentIndex()
        py_bin = self.get_python_bin()

        if idx == 0:  # 一鍵端到端 (process_book.py)
            cmd = [py_bin, self.get_script_path("process_book.py")]
            in_dir = self.oc_input.text().strip() or "samples"
            out_dir = self.oc_output.text().strip() or "output"
            cmd.extend(["--input", in_dir, "--output", out_dir])
            if self.oc_models_dir.text().strip():
                cmd.extend(["--models-dir", self.oc_models_dir.text().strip()])
            if self.oc_title.text().strip():
                cmd.extend(["--title", self.oc_title.text().strip()])
            if self.oc_cover.text().strip():
                cmd.extend(["--cover", self.oc_cover.text().strip()])
            if self.oc_dpi.text().strip():
                cmd.extend(["--dpi", self.oc_dpi.text().strip()])
            if self.oc_unsharp.text().strip():
                cmd.extend(["--unsharp", self.oc_unsharp.text().strip()])
            cmd.extend(["--workers", str(self.oc_workers.value())])
            cmd.extend(["--batch-size", str(self.oc_batch.value())])
            if self.oc_compress_images.isChecked():
                cmd.append("--compress-images")
                cmd.extend(["--image-quality", str(self.oc_img_quality.value())])
                cmd.extend(["--image-scale", str(self.oc_img_scale.value())])
                if self.oc_img_max_dim.value() > 0:
                    cmd.extend(["--image-max-dim", str(self.oc_img_max_dim.value())])
                if self.oc_img_unsharp.text().strip():
                    cmd.extend(["--image-unsharp", self.oc_img_unsharp.text().strip()])
            skips = safe_split_args(self.oc_skip.text())
            if skips:
                cmd.extend(["--skip-ocr"] + skips)
            if not self.oc_epub.isChecked():
                cmd.append("--no-epub")
            if not self.oc_organize.isChecked():
                cmd.append("--no-organize-pages")
            if self.oc_copy.isChecked():
                cmd.append("--copy")
            return cmd

        elif idx == 1:  # 步驟 1: preprocess_scantailor.py
            cmd = [py_bin, self.get_script_path("preprocess_scantailor.py")]
            in_args = safe_split_args(self.s1_input.text())
            cmd.extend(["--input"] + (in_args if in_args else ["samples"]))
            cmd.extend(["--output", self.s1_output.text().strip() or "preprocessed_samples"])
            cmd.extend(["--color-mode", self.s1_color.currentText()])
            cmd.extend(["--despeckle", self.s1_despeckle.currentText()])
            cmd.extend(["--margins", str(self.s1_margins.value())])
            cmd.extend(["--format", self.s1_format.currentText()])
            cmd.extend(["--quality", str(self.s1_quality.value())])
            cmd.extend(["--scale", str(self.s1_scale.value())])
            if self.s1_unsharp.text().strip():
                cmd.extend(["--unsharp", self.s1_unsharp.text().strip()])
            if self.s1_dpi.text().strip():
                cmd.extend(["--dpi", self.s1_dpi.text().strip()])
            s1_skips = safe_split_args(self.s1_skip.text())
            if s1_skips:
                cmd.extend(["--skip-orc"] + s1_skips)
            return cmd

        elif idx == 2:  # 步驟 2: extract_toc.py
            cmd = [py_bin, self.get_script_path("extract_toc.py")]
            in_files = safe_split_args(self.s2_input.text())
            if in_files:
                cmd.extend(["--input"] + in_files)
            cmd.extend(["--output", self.s2_output.text().strip() or "toc.json"])
            if self.s2_title.text().strip():
                cmd.extend(["--book-title", self.s2_title.text().strip()])
            if self.s2_models_dir.text().strip():
                cmd.extend(["--models-dir", self.s2_models_dir.text().strip()])
            return cmd

        elif idx == 3:  # 步驟 3: pipeline.py
            cmd = [py_bin, self.get_script_path("pipeline.py")]
            cmd.extend(["--samples", self.s3_input.text().strip() or "preprocessed_samples"])
            cmd.extend(["--output", self.s3_output.text().strip() or "output"])
            if self.s3_title.text().strip():
                cmd.extend(["--title", self.s3_title.text().strip()])
            if self.s3_toc.text().strip():
                cmd.extend(["--toc", self.s3_toc.text().strip()])
            if self.s3_cover.text().strip():
                cmd.extend(["--cover", self.s3_cover.text().strip()])
            if self.s3_models_dir.text().strip():
                cmd.extend(["--models-dir", self.s3_models_dir.text().strip()])
            cmd.extend(["--workers", str(self.s3_workers.value())])
            cmd.extend(["--batch-size", str(self.s3_batch.value())])
            if self.s3_compress_images.isChecked():
                cmd.append("--compress-images")
                cmd.extend(["--image-quality", str(self.s3_img_quality.value())])
                cmd.extend(["--image-scale", str(self.s3_img_scale.value())])
                if self.s3_img_max_dim.value() > 0:
                    cmd.extend(["--image-max-dim", str(self.s3_img_max_dim.value())])
                if self.s3_img_unsharp.text().strip():
                    cmd.extend(["--image-unsharp", self.s3_img_unsharp.text().strip()])
            skips = safe_split_args(self.s3_skip.text())
            if skips:
                cmd.extend(["--skip-ocr"] + skips)
            if self.s3_epub.isChecked():
                cmd.append("--epub")
            else:
                cmd.append("--no-epub")
            if not self.s3_cover_first.isChecked():
                cmd.append("--no-cover-first")
            return cmd

        elif idx == 4:  # 步驟 4: extract_toc.py --organize-pages
            cmd = [py_bin, self.get_script_path("extract_toc.py")]
            in_files = safe_split_args(self.s4_input.text())
            if in_files:
                cmd.extend(["--input"] + in_files)
            target_dir = self.s4_target.text().strip() or "preprocessed_samples"
            cmd.extend(["--organize-pages", target_dir])
            if self.s4_copy.isChecked():
                cmd.append("--copy")
            return cmd

        elif idx == 5:  # 步驟 5: patch_pages.py
            cmd = [py_bin, self.get_script_path("patch_pages.py")]
            cmd.extend(["--html", self.s5_html.text().strip() or "output/book.html"])
            pages = safe_split_args(self.s5_pages.text())
            if pages:
                cmd.extend(["--pages"] + pages)
            sources = safe_split_args(self.s5_source.text())
            if sources:
                cmd.extend(["--source"] + sources)
            if not self.s5_epub.isChecked():
                cmd.append("--no-epub")
            if not self.s5_backup.isChecked():
                cmd.append("--no-backup")
            return cmd

        elif idx == 6:  # 批次多書處理 (batch_process.py)
            cmd = [py_bin, self.get_script_path("batch_process.py")]
            in_dir = self.batch_input_dir.text().strip()
            if in_dir:
                cmd.extend(["--input-dir", in_dir])
            dirs = safe_split_args(self.batch_dirs.text())
            if dirs:
                cmd.extend(["--dirs"] + dirs)
            pat = self.batch_pattern.text().strip()
            if pat and pat != "*":
                cmd.extend(["--pattern", pat])
            cfg_name = self.batch_config_name.text().strip()
            if cfg_name and cfg_name != "book_config.json":
                cmd.extend(["--config-name", cfg_name])
            out_root = self.batch_output_root.text().strip() or "output_batch"
            cmd.extend(["--output-root", out_root])
            if self.batch_models_dir.text().strip():
                cmd.extend(["--models-dir", self.batch_models_dir.text().strip()])

            if self.batch_skip_existing.isChecked():
                cmd.append("--skip-existing")
            if not self.batch_use_existing_toc.isChecked():
                cmd.append("--no-use-existing-toc")
            if self.batch_force_toc.isChecked():
                cmd.append("--force-toc")
            if not self.batch_continue_on_error.isChecked():
                cmd.append("--stop-on-error")
            if self.batch_dry_run.isChecked():
                cmd.append("--dry-run")

            cmd.extend(["--workers", str(self.batch_workers.value())])
            cmd.extend(["--batch-size", str(self.batch_batch_size.value())])
            if self.batch_dpi.text().strip():
                cmd.extend(["--dpi", self.batch_dpi.text().strip()])
            if self.batch_unsharp.text().strip():
                cmd.extend(["--unsharp", self.batch_unsharp.text().strip()])
            if not self.batch_compress_images.isChecked():
                cmd.append("--no-compress-images")
            if not self.batch_epub.isChecked():
                cmd.append("--no-epub")
            if not self.batch_organize.isChecked():
                cmd.append("--no-organize-pages")
            if self.batch_copy.isChecked():
                cmd.append("--copy")
            return cmd

        return [py_bin]

    def update_command_preview(self):
        cmd = self.get_current_cmd_list()
        if not cmd:
            return
        # 友善呈現指令（將直譯器替換成 python，若腳本在當前目錄則簡化為檔名）
        script_arg = cmd[1] if len(cmd) > 1 else ""
        if script_arg:
            try:
                p = Path(script_arg)
                if p.parent == Path.cwd():
                    script_arg = p.name
            except Exception:
                pass
            display_cmd = ["python", script_arg] + cmd[2:]
        else:
            display_cmd = ["python"]

        formatted_args = []
        for arg in display_cmd:
            if " " in arg and not ((arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'"))):
                formatted_args.append(f'"{arg}"')
            else:
                formatted_args.append(arg)
        self.cmd_preview_edit.setText(" ".join(formatted_args))

    def copy_command_to_clipboard(self):
        cmd_text = self.cmd_preview_edit.text()
        QApplication.clipboard().setText(cmd_text)
        self.lbl_status.setText("已複製 CLI 指令至剪貼簿！")

    # -------------------------------------------------------------
    # JSON 偏好設定與參數範本匯入 / 匯出 (Phase 1)
    # -------------------------------------------------------------
    TAB_CONFIG_META = {
        0: {"name": "一鍵端到端總控", "default_file": "book_config.json", "tool": "process_book.py"},
        1: {"name": "步驟 1: 幾何轉正裁切", "default_file": "step1_config.json", "tool": "preprocess_scantailor.py"},
        2: {"name": "步驟 2: 目錄抽取 (TOC)", "default_file": "toc_config.json", "tool": "extract_toc.py"},
        3: {"name": "步驟 3: AI 版面與 EPUB", "default_file": "step3_config.json", "tool": "pipeline.py"},
        4: {"name": "步驟 4: 章節目錄歸檔", "default_file": "organize_config.json", "tool": "extract_toc.py"},
        5: {"name": "步驟 5: 異常頁微創修復", "default_file": "patch_config.json", "tool": "patch_pages.py"},
        6: {"name": "批次多書處理", "default_file": "batch_config.json", "tool": "batch_process.py"},
    }

    def setup_menu_bar(self):
        """建立主視窗選單列 (檔案 -> 載入/匯出偏好設定)"""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("檔案(&F)")

        act_load = file_menu.addAction("📂 載入當前分頁偏好設定 (-p)...")
        act_load.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        act_load.setStatusTip("選取並載入 JSON 偏好設定檔至目前選取的分頁")
        act_load.triggered.connect(lambda: self.action_load_preference())

        act_save = file_menu.addAction("💾 匯出當前分頁設定範本 (-t)...")
        act_save.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        act_save.setStatusTip("將目前分頁的控制項參數匯出為 JSON 設定檔")
        act_save.triggered.connect(lambda: self.action_save_template())

        file_menu.addSeparator()

        act_view_log = file_menu.addAction("📄 檢視任務詳細日誌 (scanorc_task.log)...")
        act_view_log.setShortcut(QtGui.QKeySequence("Ctrl+L"))
        act_view_log.setStatusTip("開啟當前工作目錄的 scanorc_task.log 完整日誌紀錄")
        act_view_log.triggered.connect(self.open_task_log)

        file_menu.addSeparator()

        act_exit = file_menu.addAction("離開(&X)")
        act_exit.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        act_exit.triggered.connect(self.close)

    def get_tab_config_meta(self, tab_idx: int) -> dict:
        return self.TAB_CONFIG_META.get(tab_idx, {
            "name": f"分頁 {tab_idx}",
            "default_file": f"config_tab{tab_idx}.json",
            "tool": "process_book.py"
        })

    def create_pref_toolbar(self, tab_idx: int) -> QWidget:
        """建立各步驟面板頂部之精簡偏好設定操作條"""
        meta = self.get_tab_config_meta(tab_idx)
        def_file = meta["default_file"]

        bar = QFrame()
        bar.setObjectName("prefToolbar")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(4, 2, 4, 2)
        bar_layout.setSpacing(6)

        btn_load = QPushButton("📂 載入設定 (-p)")
        btn_load.setToolTip(f"載入 JSON 偏好設定檔至此分頁 ({def_file})")
        btn_load.setStyleSheet("padding: 2px 8px; font-size: 12px;")
        btn_load.clicked.connect(lambda: self.action_load_preference(tab_idx))

        btn_save = QPushButton("💾 匯出設定 (-t)")
        btn_save.setToolTip(f"將此分頁目前的設定匯出為 JSON 設定檔 ({def_file})")
        btn_save.setStyleSheet("padding: 2px 8px; font-size: 12px;")
        btn_save.clicked.connect(lambda: self.action_save_template(tab_idx))

        lbl_info = QLabel(f"<span style='color: #666; font-size: 11px;'>偏好設定檔: <code>{def_file}</code></span>")

        bar_layout.addWidget(btn_load)
        bar_layout.addWidget(btn_save)
        bar_layout.addWidget(lbl_info)
        bar_layout.addStretch()
        return bar

    def export_tab_config(self, tab_idx: int) -> dict:
        """依據當前分頁 UI 元件數值，匯出為格式化 JSON 字典"""
        if tab_idx == 0:
            dpi_val = None
            dpi_text = self.oc_dpi.text().strip()
            if dpi_text.isdigit():
                dpi_val = int(dpi_text)

            max_dim_val = self.oc_img_max_dim.value()
            if max_dim_val <= 0:
                max_dim_val = None

            return {
                "input": self.oc_input.text().strip() or "samples",
                "output": self.oc_output.text().strip() or "output",
                "title": self.oc_title.text().strip(),
                "toc": None,
                "force_toc": False,
                "models_dir": self.oc_models_dir.text().strip() or None,
                "dpi": dpi_val,
                "margins": 1,
                "quality": 60,
                "scale": 0.8,
                "unsharp": self.oc_unsharp.text().strip(),
                "color_mode": "color_grayscale",
                "batch_size": self.oc_batch.value(),
                "workers": self.oc_workers.value(),
                "skip_ocr": safe_split_args(self.oc_skip.text()),
                "epub": self.oc_epub.isChecked(),
                "cover": self.oc_cover.text().strip() or None,
                "compress_images": self.oc_compress_images.isChecked(),
                "image_quality": self.oc_img_quality.value(),
                "image_scale": round(float(self.oc_img_scale.value()), 2),
                "image_max_dim": max_dim_val,
                "image_unsharp": self.oc_img_unsharp.text().strip() or None,
                "organize_pages": self.oc_organize.isChecked(),
                "copy": self.oc_copy.isChecked(),
                "organize_target": "preprocessed"
            }

        elif tab_idx == 1:
            dpi_val = None
            dpi_text = self.s1_dpi.text().strip()
            if dpi_text.isdigit():
                dpi_val = int(dpi_text)

            in_args = safe_split_args(self.s1_input.text())
            return {
                "input": in_args if in_args else ["samples"],
                "output": self.s1_output.text().strip() or "preprocessed_samples",
                "format": self.s1_format.currentText(),
                "dpi": dpi_val,
                "margins": self.s1_margins.value(),
                "color_mode": self.s1_color.currentText(),
                "despeckle": self.s1_despeckle.currentText(),
                "quality": self.s1_quality.value(),
                "scale": round(float(self.s1_scale.value()), 2),
                "unsharp": self.s1_unsharp.text().strip(),
                "batch_size": 0,
                "skip_ocr": safe_split_args(self.s1_skip.text())
            }

        elif tab_idx == 2:
            in_args = safe_split_args(self.s2_input.text())
            return {
                "input": in_args if in_args else [],
                "output": self.s2_output.text().strip() or "toc.json",
                "models_dir": self.s2_models_dir.text().strip() or None,
                "model_dir": "models/glm-ocr",
                "book_title": self.s2_title.text().strip(),
                "organize_pages": None,
                "copy": False
            }

        elif tab_idx == 3:
            samples = safe_split_args(self.s3_input.text())
            max_dim_val = self.s3_img_max_dim.value()
            if max_dim_val <= 0:
                max_dim_val = None

            return {
                "samples": samples if samples else ["preprocessed_samples"],
                "output": self.s3_output.text().strip() or "output",
                "title": self.s3_title.text().strip() or "數位化掃描書本",
                "toc": self.s3_toc.text().strip() or None,
                "models_dir": self.s3_models_dir.text().strip() or None,
                "epub": self.s3_epub.isChecked(),
                "cover": self.s3_cover.text().strip() or None,
                "compress_images": self.s3_compress_images.isChecked(),
                "image_quality": self.s3_img_quality.value(),
                "image_scale": round(float(self.s3_img_scale.value()), 2),
                "image_max_dim": max_dim_val,
                "image_unsharp": self.s3_img_unsharp.text().strip() or None,
                "batch_size": self.s3_batch.value(),
                "workers": self.s3_workers.value(),
                "recursive": True,
                "cover_first": self.s3_cover_first.isChecked(),
                "skip_ocr": safe_split_args(self.s3_skip.text())
            }

        elif tab_idx == 4:
            in_args = safe_split_args(self.s4_input.text())
            return {
                "input": in_args if in_args else [],
                "output": "toc.json",
                "model_dir": "models/glm-ocr",
                "book_title": "",
                "organize_pages": self.s4_target.text().strip() or "preprocessed_samples",
                "copy": self.s4_copy.isChecked()
            }

        elif tab_idx == 5:
            sources = safe_split_args(self.s5_source.text())
            return {
                "pages": safe_split_args(self.s5_pages.text()),
                "html": self.s5_html.text().strip() or "output/book.html",
                "source": sources if sources else ["output/preprocessed", "preprocessed_samples", "samples"],
                "epub": self.s5_epub.isChecked(),
                "backup": self.s5_backup.isChecked(),
                "revert": False,
                "list": False
            }

        elif tab_idx == 6:
            dpi_val = None
            dpi_text = self.batch_dpi.text().strip()
            if dpi_text.isdigit():
                dpi_val = int(dpi_text)

            dirs = safe_split_args(self.batch_dirs.text())
            in_dir = self.batch_input_dir.text().strip() or None

            return {
                "input_dir": in_dir,
                "dirs": dirs,
                "pattern": self.batch_pattern.text().strip() or "*",
                "config_name": self.batch_config_name.text().strip() or "book_config.json",
                "output_root": self.batch_output_root.text().strip() or "output_batch",
                "models_dir": self.batch_models_dir.text().strip() or None,
                "report": None,
                "skip_existing": self.batch_skip_existing.isChecked(),
                "use_existing_toc": self.batch_use_existing_toc.isChecked(),
                "force_toc": self.batch_force_toc.isChecked(),
                "continue_on_error": self.batch_continue_on_error.isChecked(),
                "stop_on_error": not self.batch_continue_on_error.isChecked(),
                "dry_run": self.batch_dry_run.isChecked(),
                "title": "",
                "cover": None,
                "dpi": dpi_val,
                "margins": 1,
                "quality": 60,
                "scale": 0.8,
                "unsharp": self.batch_unsharp.text().strip() or "1x0.55+1.5+0.002",
                "color_mode": "color_grayscale",
                "batch_size": self.batch_batch_size.value(),
                "workers": self.batch_workers.value(),
                "skip_ocr": ["Cover-*.jpg", "Content-*.jpg"],
                "epub": self.batch_epub.isChecked(),
                "compress_images": self.batch_compress_images.isChecked(),
                "image_quality": 75,
                "image_scale": 0.8,
                "image_max_dim": None,
                "organize_pages": self.batch_organize.isChecked(),
                "copy": self.batch_copy.isChecked(),
                "organize_target": "preprocessed"
            }

        return {}

    def import_tab_config(self, tab_idx: int, data: dict):
        """將 JSON 設定字典回填至指定分頁之 UI 元件中"""
        if not isinstance(data, dict):
            return

        if tab_idx == 0:
            if "input" in data and data["input"] is not None:
                val = data["input"]
                self.oc_input.setText(" ".join(val) if isinstance(val, list) else str(val))
            if "output" in data and data["output"] is not None:
                self.oc_output.setText(str(data["output"]))
            if "title" in data:
                self.oc_title.setText(str(data["title"]) if data["title"] is not None else "")
            if "cover" in data:
                self.oc_cover.setText(str(data["cover"]) if data["cover"] is not None else "")
            if "models_dir" in data:
                self.oc_models_dir.setText(str(data["models_dir"]) if data["models_dir"] is not None else "")
            if "dpi" in data:
                self.oc_dpi.setText(str(data["dpi"]) if data["dpi"] is not None else "")
            if "workers" in data and data["workers"] is not None:
                self.oc_workers.setValue(int(data["workers"]))
            if "batch_size" in data and data["batch_size"] is not None:
                self.oc_batch.setValue(int(data["batch_size"]))
            if "compress_images" in data and data["compress_images"] is not None:
                self.oc_compress_images.setChecked(bool(data["compress_images"]))
            if "image_quality" in data and data["image_quality"] is not None:
                self.oc_img_quality.setValue(int(data["image_quality"]))
            if "image_scale" in data and data["image_scale"] is not None:
                self.oc_img_scale.setValue(float(data["image_scale"]))
            if "image_max_dim" in data:
                self.oc_img_max_dim.setValue(int(data["image_max_dim"]) if data["image_max_dim"] else 0)
            if "image_unsharp" in data and data["image_unsharp"] is not None:
                self.oc_img_unsharp.setText(str(data["image_unsharp"]))
            if "unsharp" in data and data["unsharp"] is not None:
                self.oc_unsharp.setText(str(data["unsharp"]))
            if "skip_ocr" in data:
                val = data["skip_ocr"]
                if isinstance(val, list):
                    self.oc_skip.setText(" ".join(str(x) for x in val))
                elif val is not None:
                    self.oc_skip.setText(str(val))
            if "epub" in data and data["epub"] is not None:
                self.oc_epub.setChecked(bool(data["epub"]))
            if "organize_pages" in data and data["organize_pages"] is not None:
                self.oc_organize.setChecked(bool(data["organize_pages"]))
            if "copy" in data and data["copy"] is not None:
                self.oc_copy.setChecked(bool(data["copy"]))

        elif tab_idx == 1:
            if "input" in data and data["input"] is not None:
                val = data["input"]
                self.s1_input.setText(" ".join(val) if isinstance(val, list) else str(val))
            if "output" in data and data["output"] is not None:
                self.s1_output.setText(str(data["output"]))
            if "format" in data and data["format"] is not None:
                self.s1_format.setCurrentText(str(data["format"]))
            if "dpi" in data:
                self.s1_dpi.setText(str(data["dpi"]) if data["dpi"] is not None else "")
            if "margins" in data and data["margins"] is not None:
                self.s1_margins.setValue(int(data["margins"]))
            if "color_mode" in data and data["color_mode"] is not None:
                self.s1_color.setCurrentText(str(data["color_mode"]))
            if "despeckle" in data and data["despeckle"] is not None:
                self.s1_despeckle.setCurrentText(str(data["despeckle"]))
            if "quality" in data and data["quality"] is not None:
                self.s1_quality.setValue(int(data["quality"]))
            if "scale" in data and data["scale"] is not None:
                self.s1_scale.setValue(float(data["scale"]))
            if "unsharp" in data and data["unsharp"] is not None:
                self.s1_unsharp.setText(str(data["unsharp"]))
            skip_val = data.get("skip_ocr", data.get("skip_orc"))
            if skip_val is not None:
                if isinstance(skip_val, list):
                    self.s1_skip.setText(" ".join(str(x) for x in skip_val))
                else:
                    self.s1_skip.setText(str(skip_val))

        elif tab_idx == 2:
            if "input" in data and data["input"] is not None:
                val = data["input"]
                self.s2_input.setText(" ".join(val) if isinstance(val, list) else str(val))
            if "output" in data and data["output"] is not None:
                self.s2_output.setText(str(data["output"]))
            if "book_title" in data:
                self.s2_title.setText(str(data["book_title"]) if data["book_title"] is not None else "")
            if "models_dir" in data:
                self.s2_models_dir.setText(str(data["models_dir"]) if data["models_dir"] is not None else "")
            elif "model_dir" in data and data["model_dir"] and not str(data["model_dir"]).endswith("glm-ocr"):
                self.s2_models_dir.setText(str(data["model_dir"]))

        elif tab_idx == 3:
            samples_val = data.get("samples", data.get("input"))
            if samples_val is not None:
                if isinstance(samples_val, list):
                    self.s3_input.setText(" ".join(str(x) for x in samples_val))
                else:
                    self.s3_input.setText(str(samples_val))
            if "output" in data and data["output"] is not None:
                self.s3_output.setText(str(data["output"]))
            if "title" in data and data["title"] is not None:
                self.s3_title.setText(str(data["title"]))
            if "toc" in data:
                self.s3_toc.setText(str(data["toc"]) if data["toc"] is not None else "")
            if "cover" in data:
                self.s3_cover.setText(str(data["cover"]) if data["cover"] is not None else "")
            if "models_dir" in data:
                self.s3_models_dir.setText(str(data["models_dir"]) if data["models_dir"] is not None else "")
            if "workers" in data and data["workers"] is not None:
                self.s3_workers.setValue(int(data["workers"]))
            if "batch_size" in data and data["batch_size"] is not None:
                self.s3_batch.setValue(int(data["batch_size"]))
            if "compress_images" in data and data["compress_images"] is not None:
                self.s3_compress_images.setChecked(bool(data["compress_images"]))
            if "image_quality" in data and data["image_quality"] is not None:
                self.s3_img_quality.setValue(int(data["image_quality"]))
            if "image_scale" in data and data["image_scale"] is not None:
                self.s3_img_scale.setValue(float(data["image_scale"]))
            if "image_max_dim" in data:
                self.s3_img_max_dim.setValue(int(data["image_max_dim"]) if data["image_max_dim"] else 0)
            if "image_unsharp" in data and data["image_unsharp"] is not None:
                self.s3_img_unsharp.setText(str(data["image_unsharp"]))
            if "skip_ocr" in data:
                val = data["skip_ocr"]
                if isinstance(val, list):
                    self.s3_skip.setText(" ".join(str(x) for x in val))
                elif val is not None:
                    self.s3_skip.setText(str(val))
            if "epub" in data and data["epub"] is not None:
                self.s3_epub.setChecked(bool(data["epub"]))
            if "cover_first" in data and data["cover_first"] is not None:
                self.s3_cover_first.setChecked(bool(data["cover_first"]))

        elif tab_idx == 4:
            if "input" in data and data["input"] is not None:
                val = data["input"]
                self.s4_input.setText(" ".join(val) if isinstance(val, list) else str(val))
            if "organize_pages" in data and data["organize_pages"] is not None:
                self.s4_target.setText(str(data["organize_pages"]))
            if "copy" in data and data["copy"] is not None:
                self.s4_copy.setChecked(bool(data["copy"]))

        elif tab_idx == 5:
            if "html" in data and data["html"] is not None:
                self.s5_html.setText(str(data["html"]))
            if "pages" in data and data["pages"] is not None:
                val = data["pages"]
                self.s5_pages.setText(" ".join(str(x) for x in val) if isinstance(val, list) else str(val))
            if "source" in data and data["source"] is not None:
                val = data["source"]
                self.s5_source.setText(" ".join(str(x) for x in val) if isinstance(val, list) else str(val))
            if "epub" in data and data["epub"] is not None:
                self.s5_epub.setChecked(bool(data["epub"]))
            if "backup" in data and data["backup"] is not None:
                self.s5_backup.setChecked(bool(data["backup"]))

        elif tab_idx == 6:
            if "input_dir" in data:
                self.batch_input_dir.setText(str(data["input_dir"]) if data["input_dir"] is not None else "")
            if "dirs" in data and data["dirs"] is not None:
                val = data["dirs"]
                self.batch_dirs.setText(" ".join(str(x) for x in val) if isinstance(val, list) else str(val))
            if "pattern" in data and data["pattern"] is not None:
                self.batch_pattern.setText(str(data["pattern"]))
            if "config_name" in data and data["config_name"] is not None:
                self.batch_config_name.setText(str(data["config_name"]))
            if "output_root" in data and data["output_root"] is not None:
                self.batch_output_root.setText(str(data["output_root"]))
            if "models_dir" in data:
                self.batch_models_dir.setText(str(data["models_dir"]) if data["models_dir"] is not None else "")
            if "skip_existing" in data and data["skip_existing"] is not None:
                self.batch_skip_existing.setChecked(bool(data["skip_existing"]))
            if "use_existing_toc" in data and data["use_existing_toc"] is not None:
                self.batch_use_existing_toc.setChecked(bool(data["use_existing_toc"]))
            if "force_toc" in data and data["force_toc"] is not None:
                self.batch_force_toc.setChecked(bool(data["force_toc"]))
            if "continue_on_error" in data and data["continue_on_error"] is not None:
                self.batch_continue_on_error.setChecked(bool(data["continue_on_error"]))
            elif "stop_on_error" in data and data["stop_on_error"] is not None:
                self.batch_continue_on_error.setChecked(not bool(data["stop_on_error"]))
            if "dry_run" in data and data["dry_run"] is not None:
                self.batch_dry_run.setChecked(bool(data["dry_run"]))
            if "workers" in data and data["workers"] is not None:
                self.batch_workers.setValue(int(data["workers"]))
            if "batch_size" in data and data["batch_size"] is not None:
                self.batch_batch_size.setValue(int(data["batch_size"]))
            if "dpi" in data:
                self.batch_dpi.setText(str(data["dpi"]) if data["dpi"] is not None else "")
            if "unsharp" in data and data["unsharp"] is not None:
                self.batch_unsharp.setText(str(data["unsharp"]))
            if "compress_images" in data and data["compress_images"] is not None:
                self.batch_compress_images.setChecked(bool(data["compress_images"]))
            if "epub" in data and data["epub"] is not None:
                self.batch_epub.setChecked(bool(data["epub"]))
            if "organize_pages" in data and data["organize_pages"] is not None:
                self.batch_organize.setChecked(bool(data["organize_pages"]))
            if "copy" in data and data["copy"] is not None:
                self.batch_copy.setChecked(bool(data["copy"]))

        self.update_command_preview()

    def action_load_preference(self, tab_idx: Optional[int] = None):
        """開啟檔案選擇對話框，讀取 JSON 偏好設定檔並套用至指定分頁"""
        if tab_idx is None:
            tab_idx = self.tabs.currentIndex()
        meta = self.get_tab_config_meta(tab_idx)
        default_path = Path.cwd() / meta["default_file"]

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"載入偏好設定 (-p) - {meta['name']}",
            str(default_path if default_path.exists() else Path.cwd()),
            "JSON 設定檔 (*.json);;所有檔案 (*.*)"
        )
        if not file_path:
            return

        try:
            data = load_json_with_comments(file_path)
            if not isinstance(data, dict):
                QMessageBox.warning(self, "格式不符", f"JSON 檔案頂層必須是鍵值物件 (dict)，目前為: {type(data).__name__}")
                return
            self.import_tab_config(tab_idx, data)
            rel_name = format_rel_path(file_path)
            self.lbl_status.setText(f"已成功載入分頁偏好設定: {rel_name}")
        except Exception as e:
            QMessageBox.critical(self, "讀取設定失敗", f"無法解析 JSON 設定檔:\n{e}")

    def action_save_template(self, tab_idx: Optional[int] = None):
        """開啟存檔對話框，將指定分頁目前的 UI 控制項數值匯出為 JSON 設定檔"""
        if tab_idx is None:
            tab_idx = self.tabs.currentIndex()
        meta = self.get_tab_config_meta(tab_idx)
        default_path = Path.cwd() / meta["default_file"]

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"匯出偏好設定 (-t) - {meta['name']}",
            str(default_path),
            "JSON 設定檔 (*.json);;所有檔案 (*.*)"
        )
        if not file_path:
            return

        try:
            config_data = self.export_tab_config(tab_idx)
            json_text = json.dumps(config_data, indent=2, ensure_ascii=False)
            Path(file_path).write_text(json_text + "\n", encoding="utf-8")
            rel_name = format_rel_path(file_path)
            self.lbl_status.setText(f"已成功匯出分頁設定檔: {rel_name}")
        except Exception as e:
            QMessageBox.critical(self, "匯出設定失敗", f"無法儲存 JSON 設定檔:\n{e}")

    # -------------------------------------------------------------
    # 進程執行與通訊
    # -------------------------------------------------------------
    def run_current_tab_step(self):
        cmd = self.get_current_cmd_list()
        self.execute_cmd_list(cmd)

    def open_task_log(self):
        log_path = Path.cwd() / "scanorc_task.log"
        if log_path.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(log_path.resolve())))
        else:
            QMessageBox.information(self, "提示", f"目前尚未產生任務日誌檔案：\n{log_path}\n請先執行任一步驟或批次任務。")

    def append_task_log(self, text: str):
        try:
            log_file = Path.cwd() / "scanorc_task.log"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

    def execute_cmd_list(self, cmd: List[str]):
        if self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "忙碌中", "目前已有處理任務正在執行，請等待其完成或點擊中斷！")
            return

        cmd_display = " ".join(["python"] + cmd[1:])
        start_banner = (
            f"\n{'=' * 60}\n"
            f"[*] ScanORC 任務啟動時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"[*] 執行直譯器: {cmd[0]}\n"
            f"[*] 調度指令: {cmd_display}\n"
            f"[*] 工作目錄: {Path.cwd()}\n"
            f"[*] 詳細日誌: scanorc_task.log\n"
            f"{'=' * 60}\n"
        )
        self.log_text.append(start_banner)
        self.append_task_log(start_banner)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        script_name = Path(cmd[1]).name if len(cmd) > 1 else "任務"
        self.lbl_status.setText(f"執行中: {script_name} ...")

        # 為 Python 直譯器注入 -u 參數與環境變數 PYTHONUNBUFFERED=1，確保所有 stdout/stderr 毫秒級無緩衝刷新
        real_cmd = list(cmd)
        if len(real_cmd) >= 2 and ("python" in Path(real_cmd[0]).name):
            if "-u" not in real_cmd[1:2]:
                real_cmd.insert(1, "-u")

        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        py_bin_path = Path(real_cmd[0])
        if py_bin_path.is_file() and py_bin_path.parent.is_dir():
            curr_path = env.value("PATH", "")
            bin_dir = str(py_bin_path.parent)
            if bin_dir not in curr_path:
                env.insert("PATH", f"{bin_dir}:{curr_path}")
        self.process.setProcessEnvironment(env)

        self.process.setWorkingDirectory(str(Path.cwd()))
        self.process.start(real_cmd[0], real_cmd[1:])

    def _read_process_stream(self, data):
        if not data:
            return
        text = bytes(data).decode("utf-8", errors="replace")
        clean_text = strip_ansi(text)
        self.log_text.insertPlainText(clean_text)
        self.append_task_log(clean_text)
        cursor = self.log_text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    def on_process_output(self):
        self._read_process_stream(self.process.readAllStandardOutput())

    def on_process_error_output(self):
        self._read_process_stream(self.process.readAllStandardError())

    def on_process_error(self, err):
        err_names = {
            getattr(getattr(QProcess, 'ProcessError', None), 'FailedToStart', 0): "無法啟動進程 (直譯器或檔案不存在，或權限不足)",
            getattr(getattr(QProcess, 'ProcessError', None), 'Crashed', 1): "進程異常崩潰或被系統信號強制終止",
            getattr(getattr(QProcess, 'ProcessError', None), 'Timedout', 2): "進程執行超時",
            getattr(getattr(QProcess, 'ProcessError', None), 'WriteError', 3): "寫入進程失敗",
            getattr(getattr(QProcess, 'ProcessError', None), 'ReadError', 4): "讀取進程輸出失敗",
            getattr(getattr(QProcess, 'ProcessError', None), 'UnknownError', 5): "未知進程錯誤",
        }
        name = err_names.get(err, f"錯誤代碼 {err}")
        err_msg = f"\n[-] QProcess 系統層事件: {name}\n"
        self.log_text.append(err_msg)
        self.append_task_log(err_msg)

    def on_process_finished(self, exit_code, exit_status):
        # 1. 確保剩餘緩衝區全數刷新輸出至終端機與日誌
        self._read_process_stream(self.process.readAllStandardOutput())
        self._read_process_stream(self.process.readAllStandardError())

        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)

        # 2. 判讀 Exit Status 與 Signal
        crash_enum = getattr(getattr(QProcess, 'ExitStatus', None), 'CrashExit', None)
        is_crash = (exit_status == crash_enum) if crash_enum is not None else False

        signal_hints = {
            15: "SIGTERM: 進程接收到終止信號 (可能為點擊「中斷執行」按鈕，或系統顯存/記憶體不足被守護行程/OOM 終止)",
            9: "SIGKILL: 進程被強制終止 (常見於系統記憶體嚴重不足被 Linux OOM-Killer 強制殺除)",
            11: "SIGSEGV: 記憶體段錯誤 (Segmentation Fault，常見於底層 C/C++ 函式庫或 GPU 驅動異常)",
            6: "SIGABRT: 進程異常中止 (Aborted)",
            2: "SIGINT: 中斷信號 (Ctrl+C 中斷)",
            1: "通用錯誤 (請向上捲動終端機檢視詳細 Python 例外與 Traceback，或檢視 scanorc_task.log)",
            2: "命令列參數無效或引數錯誤",
        }
        detail_msg = signal_hints.get(exit_code, f"狀態代碼: {exit_code}")

        if exit_code == 0 and not is_crash:
            self.lbl_status.setText("✅ 執行成功完成！")
            msg = "\n[+] 任務順利完成 (結束碼: 0)"
            self.log_text.append(msg)
            self.append_task_log(msg + "\n")
            if self.tabs.currentIndex() == 2:
                self.load_toc_json()
        else:
            self.lbl_status.setText(f"❌ 執行中斷或異常 (代碼: {exit_code})")
            err_log = (
                f"\n==================================================\n"
                f"[-] 任務異常結束！\n"
                f"    - 退出代碼: {exit_code}\n"
                f"    - 診斷提示: {detail_msg}\n"
                f"    - 崩潰狀態: {'是 (CrashExit)' if is_crash else '否 (NormalExit)'}\n"
                f"    - 完整日誌已同步儲存於: scanorc_task.log\n"
                f"=================================================="
            )
            self.log_text.append(err_log)
            self.append_task_log(err_log + "\n")

    def stop_current_process(self):
        if self.process.state() != QProcess.NotRunning:
            self.lbl_status.setText("正在終止進程...")
            stop_msg = "\n[*] 使用者發起中斷執行請求 (發送 SIGTERM 終止信號)...\n"
            self.log_text.append(stop_msg)
            self.append_task_log(stop_msg)
            self.process.terminate()
            QtCore.QTimer.singleShot(2500, self.force_kill_process_if_alive)

    def force_kill_process_if_alive(self):
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()
            self.lbl_status.setText("已強制終止進程！")

    # -------------------------------------------------------------
    # 捷徑開啟
    # -------------------------------------------------------------
    def open_generated_html(self):
        html_p = Path(self.s3_output.text().strip() or "output") / "book.html"
        if html_p.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(html_p.resolve())))
        else:
            QMessageBox.information(self, "提示", f"尚未生成 HTML: {html_p}")

    def open_generated_epub(self):
        epub_p = Path(self.s3_output.text().strip() or "output") / "book.epub"
        if epub_p.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(epub_p.resolve())))
        else:
            QMessageBox.information(self, "提示", f"尚未生成 EPUB: {epub_p}")

    def rebuild_epub_from_html(self):
        html_p = Path(self.s3_output.text().strip() or "output") / "book.html"
        if not html_p.exists():
            QMessageBox.warning(self, "提示", f"尚未生成 HTML 檔案: {html_p}\n請先確認該檔案存在或已執行步驟 3。")
            return
        cmd = [self.get_python_bin(), self.get_script_path("html_to_epub.py"), str(html_p)]
        if self.s3_title.text().strip():
            cmd.extend(["--title", self.s3_title.text().strip()])
        if self.s3_toc.text().strip():
            cmd.extend(["--toc", self.s3_toc.text().strip()])
        if self.s3_cover.text().strip():
            cmd.extend(["--cover", self.s3_cover.text().strip()])
        if self.s3_compress_images.isChecked():
            cmd.append("--compress-images")
            cmd.extend(["--image-quality", str(self.s3_img_quality.value())])
            cmd.extend(["--image-scale", str(self.s3_img_scale.value())])
            if self.s3_img_max_dim.value() > 0:
                cmd.extend(["--image-max-dim", str(self.s3_img_max_dim.value())])
            if self.s3_img_unsharp.text().strip():
                cmd.extend(["--image-unsharp", self.s3_img_unsharp.text().strip()])
        self.execute_cmd_list(cmd)

    def open_folder(self, folder_str: str):
        p = Path(folder_str.strip() or ".")
        if p.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(p.resolve())))
        else:
            QMessageBox.information(self, "提示", f"目錄不存在: {p}")

    def apply_styles(self):
        """套用現代乾淨桌面風格與 Emoji/CJK 字型鏈"""
        ui_font_stack = get_ui_font_family_str()
        self.setStyleSheet(f"""
            * {{
                font-family: {ui_font_stack};
            }}
            QMainWindow {{
                background-color: #f7f9fa;
            }}
            QTabWidget::pane {{
                border: 1px solid #d0d7de;
                background: white;
                border-radius: 4px;
            }}
            QTabBar::tab {{
                font-family: {ui_font_stack};
                background: #eef2f5;
                border: 1px solid #d0d7de;
                padding: 8px 14px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: 500;
            }}
            QTabBar::tab:selected {{
                background: white;
                border-bottom-color: white;
                font-weight: bold;
                color: #007acc;
            }}
            QGroupBox {{
                font-weight: bold;
                border: 1px solid #d0d7de;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                background: #ffffff;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
            QPushButton {{
                font-family: {ui_font_stack};
                border-radius: 4px;
                padding: 4px 10px;
                border: 1px solid #bbb;
                background-color: #f6f8fa;
            }}
            QPushButton:hover {{
                background-color: #e6ebf1;
            }}
            QTableWidget {{
                font-family: {ui_font_stack};
                border: 1px solid #d0d7de;
                background-color: white;
                gridline-color: #f0f0f0;
            }}
            QHeaderView::section {{
                font-family: {ui_font_stack};
                background-color: #f0f3f6;
                padding: 4px;
                border: 1px solid #e1e4e8;
                font-weight: bold;
            }}
            QLineEdit, QSpinBox, QComboBox {{
                font-family: {ui_font_stack};
                padding: 4px;
                border: 1px solid #ccc;
                border-radius: 3px;
                background: white;
            }}
            QFrame#prefToolbar {{
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 4px;
                padding: 1px 4px;
            }}
        """)

    def apply_terminal_style(self):
        """套用終端機視窗背景、前景顏色與字體大小"""
        theme = TERMINAL_THEMES.get(self.current_theme_key, TERMINAL_THEMES["dark"])
        bg = theme["bg"]
        fg = theme["fg"]
        sel_bg = theme["selection_bg"]
        fs = self.current_font_size

        font_stack = "'JetBrains Mono', 'Fira Code', 'DejaVu Sans Mono', 'Ubuntu Mono', 'Consolas', 'Courier New', monospace"
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {bg};
                color: {fg};
                selection-background-color: {sel_bg};
                selection-color: {fg};
                font-family: {font_stack};
                font-size: {fs}px;
                line-height: 1.4;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 6px;
            }}
        """)
        if hasattr(self, "highlighter") and self.highlighter:
            self.highlighter.set_theme(self.current_theme_key)

    def on_terminal_theme_changed(self, index: int):
        theme_key = self.combo_theme.itemData(index)
        if theme_key and theme_key != self.current_theme_key:
            self.current_theme_key = theme_key
            self.apply_terminal_style()

    def on_terminal_font_size_changed(self, index: int):
        size = self.combo_font_size.itemData(index)
        if size and size != self.current_font_size:
            self.current_font_size = size
            self.apply_terminal_style()

    def on_terminal_zoom(self, delta_step: int):
        """響應 Ctrl + 滾輪或按鈕縮放"""
        idx = self.combo_font_size.currentIndex()
        new_idx = max(0, min(self.combo_font_size.count() - 1, idx + delta_step))
        if new_idx != idx:
            self.combo_font_size.setCurrentIndex(new_idx)


def main():
    app = QApplication(sys.argv)
    setup_app_fonts(app)
    app.setApplicationName("ScanORC GUI")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
