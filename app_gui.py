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

# -------------------------------------------------------------
# Linux / Conda 環境相容性自動引導 (解決 Conda Qt5 插件衝突、舊版 libxkbcommon 段錯誤與 libxcb-cursor 缺失)
# -------------------------------------------------------------
project_root = Path(__file__).resolve().parent
bundled_lib_dir = project_root / "bin" / "lib"

# 檢查是否需要透過 LD_PRELOAD 解決 Conda 舊版 libxkbcommon 造成的段錯誤
if sys.platform.startswith("linux") and not os.environ.get("_SCANORC_GUI_BOOTSTRAPPED"):
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
        os.execv(sys.executable, [sys.executable] + sys.argv)

try:
    import PySide6
    # 1. 強制設定 QT_PLUGIN_PATH 指向 PySide6 自帶的 Qt6 外掛目錄，避免 Conda 載入舊版 Qt5 插件
    pyside_dir = Path(PySide6.__file__).resolve().parent
    qt_plugins_dir = pyside_dir / "Qt" / "plugins"
    if qt_plugins_dir.exists():
        os.environ["QT_PLUGIN_PATH"] = str(qt_plugins_dir)

    # 2. 自動補充 PySide6 Qt/lib 缺失之函式庫
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
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QFileDialog, QMessageBox,
    QGroupBox, QFormLayout, QProgressBar, QFrame
)

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
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScanORC - 掃描書本轉語意結構化 HTML/EPUB 控制台")
        self.resize(1280, 850)

        # 終端機色彩與字體狀態
        self.current_theme_key = "dark"
        self.current_font_size = 14
        self.available_font_sizes = [11, 12, 13, 14, 15, 16, 18, 20, 22, 24]

        # QProcess 實例
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.on_process_output)
        self.process.finished.connect(self.on_process_finished)

        self.setup_ui()
        self.apply_styles()

        # 預設自動偵測載入 samples 目錄
        if Path("samples").exists():
            self.file_manager.add_paths(list(Path("samples").glob("*.jpg")))

    def setup_ui(self):
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
        self.file_manager.files_updated.connect(self.on_files_updated)
        self.file_manager.selection_changed.connect(self.on_file_selection_changed)
        self.h_splitter.addWidget(self.file_manager)

        # 2. 右側步驟頁籤
        self.tabs = QTabWidget()
        self.h_splitter.addWidget(self.tabs)
        self.h_splitter.setStretchFactor(0, 1)
        self.h_splitter.setStretchFactor(1, 2)

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

        self.tabs.addTab(self.tab_oneclick, "🚀 一鍵端到端總控")
        self.tabs.addTab(self.tab_step1, "📐 步驟 1: 幾何轉正裁切")
        self.tabs.addTab(self.tab_step2, "📑 步驟 2: 目錄抽取 (TOC)")
        self.tabs.addTab(self.tab_step3, "🤖 步驟 3: AI 版面與 EPUB")
        self.tabs.addTab(self.tab_step4, "📁 步驟 4: 章節目錄歸檔")
        self.tabs.addTab(self.tab_step5, "🩹 步驟 5: 異常頁微創修復")

        self.init_tab_oneclick()
        self.init_tab_step1()
        self.init_tab_step2()
        self.init_tab_step3()
        self.init_tab_step4()
        self.init_tab_step5()

    # -------------------------------------------------------------
    # Tab 0: 一鍵端到端總控 (process_book.py)
    # -------------------------------------------------------------
    def init_tab_oneclick(self):
        layout = QVBoxLayout(self.tab_oneclick)
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
        self.oc_cover.setPlaceholderText("留空則自動偵測 Cover-*.jpg 或 cover.*")
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

        h_opts = QHBoxLayout()
        self.oc_workers = QSpinBox()
        self.oc_workers.setRange(1, 16)
        self.oc_workers.setValue(4)
        self.oc_workers.valueChanged.connect(self.update_command_preview)
        h_opts.addWidget(QLabel("並行 Workers:"))
        h_opts.addWidget(self.oc_workers)

        self.oc_batch = QSpinBox()
        self.oc_batch.setRange(1, 8)
        self.oc_batch.setValue(4)
        self.oc_batch.valueChanged.connect(self.update_command_preview)
        h_opts.addWidget(QLabel("單頁 Batch:"))
        h_opts.addWidget(self.oc_batch)

        self.oc_dpi = QLineEdit("")
        self.oc_dpi.setPlaceholderText("留空表示 1:1 原生尺寸")
        self.oc_dpi.textChanged.connect(self.update_command_preview)
        h_opts.addWidget(QLabel("DPI:"))
        h_opts.addWidget(self.oc_dpi)
        form.addRow("效能與尺寸參數:", h_opts)

        # images 插圖重新壓縮與尺寸縮放設定
        self.oc_compress_images = QCheckBox("啟用 images 插圖壓縮與縮放 (--compress-images)")
        self.oc_compress_images.setChecked(False)
        self.oc_compress_images.stateChanged.connect(self.update_command_preview)

        self.oc_img_quality = QSpinBox()
        self.oc_img_quality.setRange(1, 100)
        self.oc_img_quality.setValue(75)
        self.oc_img_quality.setSuffix(" %")
        self.oc_img_quality.valueChanged.connect(self.update_command_preview)

        self.oc_img_scale = QDoubleSpinBox()
        self.oc_img_scale.setRange(0.1, 2.0)
        self.oc_img_scale.setSingleStep(0.05)
        self.oc_img_scale.setValue(1.0)
        self.oc_img_scale.valueChanged.connect(self.update_command_preview)

        self.oc_img_max_dim = QSpinBox()
        self.oc_img_max_dim.setRange(0, 4096)
        self.oc_img_max_dim.setValue(0)
        self.oc_img_max_dim.setSpecialValueText("不限制 (原尺寸)")
        self.oc_img_max_dim.valueChanged.connect(self.update_command_preview)

        h_img_opt = QHBoxLayout()
        h_img_opt.addWidget(QLabel("品質:"))
        h_img_opt.addWidget(self.oc_img_quality)
        h_img_opt.addWidget(QLabel("縮放:"))
        h_img_opt.addWidget(self.oc_img_scale)
        h_img_opt.addWidget(QLabel("最大長寬:"))
        h_img_opt.addWidget(self.oc_img_max_dim)

        form.addRow("images 圖片優化:", self.oc_compress_images)
        form.addRow("圖片品質與尺寸:", h_img_opt)

        self.oc_skip = QLineEdit("")
        self.oc_skip.setPlaceholderText("自動同步檔案清單之『略過OCR』勾選項，亦可在此手動輸入")
        self.oc_skip.textChanged.connect(self.update_command_preview)
        form.addRow("略過 OCR 頁面 (--skip-ocr):", self.oc_skip)

        h_chk = QHBoxLayout()
        self.oc_epub = QCheckBox("自動編譯 MathML EPUB3 (--epub)")
        self.oc_epub.setChecked(True)
        self.oc_epub.stateChanged.connect(self.update_command_preview)
        h_chk.addWidget(self.oc_epub)

        self.oc_copy = QCheckBox("章節歸檔使用複製而非移動 (--copy)")
        self.oc_copy.stateChanged.connect(self.update_command_preview)
        h_chk.addWidget(self.oc_copy)
        form.addRow("功能開關:", h_chk)

        layout.addLayout(form)
        desc = QLabel(
            "<b>說明：</b> 一鍵自動串聯【轉正裁切 $\\rightarrow$ 目錄抽取 $\\rightarrow$ AI OCR & EPUB $\\rightarrow$ 章節歸檔】全流程。"
        )
        desc.setStyleSheet("color: #444; margin-top: 10px;")
        layout.addWidget(desc)
        layout.addStretch()

    # -------------------------------------------------------------
    # Tab 1: 幾何預處理 (preprocess_scantailor.py)
    # -------------------------------------------------------------
    def init_tab_step1(self):
        layout = QVBoxLayout(self.tab_step1)
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
        self.s1_margins.setValue(10)
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
        self.s1_quality.setValue(95)
        self.s1_quality.valueChanged.connect(self.update_command_preview)
        h_fmt.addWidget(QLabel("品質:"))
        h_fmt.addWidget(self.s1_quality)

        self.s1_dpi = QLineEdit("")
        self.s1_dpi.setPlaceholderText("留空為 1:1 原生尺寸")
        self.s1_dpi.textChanged.connect(self.update_command_preview)
        h_fmt.addWidget(QLabel("DPI:"))
        h_fmt.addWidget(self.s1_dpi)
        form.addRow("輸出品質:", h_fmt)

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
        desc.setStyleSheet("color: #444; margin-top: 10px;")
        layout.addWidget(desc)
        layout.addStretch()

    # -------------------------------------------------------------
    # Tab 2: 抽取目錄結構 (extract_toc.py)
    # -------------------------------------------------------------
    def init_tab_step2(self):
        layout = QVBoxLayout(self.tab_step2)
        form = QFormLayout()

        self.s2_input = QLineEdit("preprocessed_samples/Cover-06.jpg")
        self.s2_input.setPlaceholderText("輸入目錄圖檔，可指定多個檔案 (如 Cover-06.jpg Cover-07.jpg)")
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
        self.s3_cover.setPlaceholderText("留空則自動偵測 Cover-*.jpg 或 cover.*")
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

        h_perf = QHBoxLayout()
        self.s3_workers = QSpinBox()
        self.s3_workers.setRange(1, 16)
        self.s3_workers.setValue(4)
        self.s3_workers.valueChanged.connect(self.update_command_preview)
        h_perf.addWidget(QLabel("Workers 並行數:"))
        h_perf.addWidget(self.s3_workers)

        self.s3_batch = QSpinBox()
        self.s3_batch.setRange(1, 8)
        self.s3_batch.setValue(4)
        self.s3_batch.valueChanged.connect(self.update_command_preview)
        h_perf.addWidget(QLabel("單頁 Batch:"))
        h_perf.addWidget(self.s3_batch)
        form.addRow("推理並行設定:", h_perf)

        # images 插圖重新壓縮與尺寸縮放設定
        self.s3_compress_images = QCheckBox("啟用 images 插圖壓縮與縮放 (--compress-images)")
        self.s3_compress_images.setChecked(False)
        self.s3_compress_images.stateChanged.connect(self.update_command_preview)

        self.s3_img_quality = QSpinBox()
        self.s3_img_quality.setRange(1, 100)
        self.s3_img_quality.setValue(75)
        self.s3_img_quality.setSuffix(" %")
        self.s3_img_quality.valueChanged.connect(self.update_command_preview)

        self.s3_img_scale = QDoubleSpinBox()
        self.s3_img_scale.setRange(0.1, 2.0)
        self.s3_img_scale.setSingleStep(0.05)
        self.s3_img_scale.setValue(1.0)
        self.s3_img_scale.valueChanged.connect(self.update_command_preview)

        self.s3_img_max_dim = QSpinBox()
        self.s3_img_max_dim.setRange(0, 4096)
        self.s3_img_max_dim.setValue(0)
        self.s3_img_max_dim.setSpecialValueText("不限制 (原尺寸)")
        self.s3_img_max_dim.valueChanged.connect(self.update_command_preview)

        h_s3_img_opt = QHBoxLayout()
        h_s3_img_opt.addWidget(QLabel("品質:"))
        h_s3_img_opt.addWidget(self.s3_img_quality)
        h_s3_img_opt.addWidget(QLabel("縮放:"))
        h_s3_img_opt.addWidget(self.s3_img_scale)
        h_s3_img_opt.addWidget(QLabel("最大長寬:"))
        h_s3_img_opt.addWidget(self.s3_img_max_dim)

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
        h_q = QHBoxLayout(box_quick)
        btn_open_html = QPushButton("🌐 在瀏覽器開啟 book.html")
        btn_open_epub = QPushButton("📖 開啟 book.epub")
        btn_rebuild_epub = QPushButton("⚡ 從 HTML 直接重編 EPUB")
        btn_rebuild_epub.setToolTip("調用 html_to_epub.py，免跑 AI/OCR 於 1~2 秒內直接將 HTML 與 images 打包為最新 EPUB")
        btn_open_out = QPushButton("📂 開啟 output 目錄")
        btn_open_html.clicked.connect(self.open_generated_html)
        btn_open_epub.clicked.connect(self.open_generated_epub)
        btn_rebuild_epub.clicked.connect(self.rebuild_epub_from_html)
        btn_open_out.clicked.connect(lambda: self.open_folder(self.s3_output.text()))
        h_q.addWidget(btn_open_html)
        h_q.addWidget(btn_open_epub)
        h_q.addWidget(btn_rebuild_epub)
        h_q.addWidget(btn_open_out)
        layout.addWidget(box_quick)

        layout.addStretch()

    # -------------------------------------------------------------
    # Tab 4: 章節目錄歸檔 (extract_toc.py --organize-pages)
    # -------------------------------------------------------------
    def init_tab_step4(self):
        layout = QVBoxLayout(self.tab_step4)
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

        self.s4_input = QLineEdit("preprocessed_samples/Cover-06.jpg")
        self.s4_input.setPlaceholderText("目錄圖檔 (Cover-xx.jpg) 或 toc.json")
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
        desc.setStyleSheet("color: #444; margin-top: 10px;")
        layout.addWidget(desc)
        layout.addStretch()

    # -------------------------------------------------------------
    # Tab 5: 頁面微創替換與校對 (patch_pages.py)
    # -------------------------------------------------------------
    def init_tab_step5(self):
        layout = QVBoxLayout(self.tab_step5)
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
        desc.setStyleSheet("color: #444; margin-top: 10px;")
        layout.addWidget(desc)
        layout.addStretch()

    def browse_patch_html(self):
        f, _ = QFileDialog.getOpenFileName(self, "選取 HTML 檔案", str(Path.cwd()), "HTML Files (*.html)")
        if f:
            self.s5_html.setText(f)

    def run_patch_list(self):
        html_file = self.s5_html.text().strip() or "output/book.html"
        cmd = [self.get_python_bin(), "patch_pages.py", "--html", html_file, "--list"]
        self.execute_cmd_list(cmd)

    def run_patch_revert(self):
        html_file = self.s5_html.text().strip() or "output/book.html"
        ret = QMessageBox.question(self, "確認還原", f"確定要從備份檔還原 {html_file} 嗎？", QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            cmd = [self.get_python_bin(), "patch_pages.py", "--html", html_file, "--revert"]
            if not self.s5_epub.isChecked():
                cmd.append("--no-epub")
            self.execute_cmd_list(cmd)

    # -------------------------------------------------------------
    # 輔助與事件處理
    # -------------------------------------------------------------
    def get_python_bin(self) -> str:
        """取得適當的 Python 直譯器 (優先使用包含依賴的專案內置 .venv)"""
        venv_py = Path.cwd() / ".venv" / "bin" / "python"
        if venv_py.exists():
            return str(venv_py)
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
        # a. 待歸檔圖檔目錄 (--organize-pages): 填入所選檔案所在目錄
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
            if not self.s4_target.text() or self.s4_target.text() == "preprocessed_samples":
                self.s4_target.setText(formatted_first_dir)

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
            cmd = [py_bin, "process_book.py"]
            in_dir = self.oc_input.text().strip() or "samples"
            out_dir = self.oc_output.text().strip() or "output"
            cmd.extend(["--input", in_dir, "--output", out_dir])
            if self.oc_title.text().strip():
                cmd.extend(["--title", self.oc_title.text().strip()])
            if self.oc_cover.text().strip():
                cmd.extend(["--cover", self.oc_cover.text().strip()])
            if self.oc_dpi.text().strip():
                cmd.extend(["--dpi", self.oc_dpi.text().strip()])
            cmd.extend(["--workers", str(self.oc_workers.value())])
            cmd.extend(["--batch-size", str(self.oc_batch.value())])
            if self.oc_compress_images.isChecked():
                cmd.append("--compress-images")
                cmd.extend(["--image-quality", str(self.oc_img_quality.value())])
                cmd.extend(["--image-scale", str(self.oc_img_scale.value())])
                if self.oc_img_max_dim.value() > 0:
                    cmd.extend(["--image-max-dim", str(self.oc_img_max_dim.value())])
            skips = safe_split_args(self.oc_skip.text())
            if skips:
                cmd.extend(["--skip-ocr"] + skips)
            if not self.oc_epub.isChecked():
                cmd.append("--no-epub")
            if self.oc_copy.isChecked():
                cmd.append("--copy")
            return cmd

        elif idx == 1:  # 步驟 1: preprocess_scantailor.py
            cmd = [py_bin, "preprocess_scantailor.py"]
            in_args = safe_split_args(self.s1_input.text())
            cmd.extend(["--input"] + (in_args if in_args else ["samples"]))
            cmd.extend(["--output", self.s1_output.text().strip() or "preprocessed_samples"])
            cmd.extend(["--color-mode", self.s1_color.currentText()])
            cmd.extend(["--despeckle", self.s1_despeckle.currentText()])
            cmd.extend(["--margins", str(self.s1_margins.value())])
            cmd.extend(["--format", self.s1_format.currentText()])
            cmd.extend(["--quality", str(self.s1_quality.value())])
            if self.s1_dpi.text().strip():
                cmd.extend(["--dpi", self.s1_dpi.text().strip()])
            s1_skips = safe_split_args(self.s1_skip.text())
            if s1_skips:
                cmd.extend(["--skip-orc"] + s1_skips)
            return cmd

        elif idx == 2:  # 步驟 2: extract_toc.py
            cmd = [py_bin, "extract_toc.py"]
            in_files = safe_split_args(self.s2_input.text())
            if in_files:
                cmd.extend(["--input"] + in_files)
            cmd.extend(["--output", self.s2_output.text().strip() or "toc.json"])
            if self.s2_title.text().strip():
                cmd.extend(["--book-title", self.s2_title.text().strip()])
            return cmd

        elif idx == 3:  # 步驟 3: pipeline.py
            cmd = [py_bin, "pipeline.py"]
            cmd.extend(["--samples", self.s3_input.text().strip() or "preprocessed_samples"])
            cmd.extend(["--output", self.s3_output.text().strip() or "output"])
            if self.s3_title.text().strip():
                cmd.extend(["--title", self.s3_title.text().strip()])
            if self.s3_toc.text().strip():
                cmd.extend(["--toc", self.s3_toc.text().strip()])
            if self.s3_cover.text().strip():
                cmd.extend(["--cover", self.s3_cover.text().strip()])
            cmd.extend(["--workers", str(self.s3_workers.value())])
            cmd.extend(["--batch-size", str(self.s3_batch.value())])
            if self.s3_compress_images.isChecked():
                cmd.append("--compress-images")
                cmd.extend(["--image-quality", str(self.s3_img_quality.value())])
                cmd.extend(["--image-scale", str(self.s3_img_scale.value())])
                if self.s3_img_max_dim.value() > 0:
                    cmd.extend(["--image-max-dim", str(self.s3_img_max_dim.value())])
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
            cmd = [py_bin, "extract_toc.py"]
            in_files = safe_split_args(self.s4_input.text())
            if in_files:
                cmd.extend(["--input"] + in_files)
            target_dir = self.s4_target.text().strip()
            if target_dir:
                cmd.extend(["--organize-pages", target_dir])
            else:
                cmd.append("--organize-pages")
            if self.s4_copy.isChecked():
                cmd.append("--copy")
            return cmd

        elif idx == 5:  # 步驟 5: patch_pages.py
            cmd = [py_bin, "patch_pages.py"]
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

        return [py_bin]

    def update_command_preview(self):
        cmd = self.get_current_cmd_list()
        # 友善呈現指令（將直譯器替換成 python）
        display_cmd = ["python"] + cmd[1:]
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
    # 進程執行與通訊
    # -------------------------------------------------------------
    def run_current_tab_step(self):
        cmd = self.get_current_cmd_list()
        self.execute_cmd_list(cmd)

    def execute_cmd_list(self, cmd: List[str]):
        if self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "忙碌中", "目前已有處理任務正在執行，請等待其完成或點擊中斷！")
            return

        cmd_display = " ".join(["python"] + cmd[1:])
        self.log_text.append(f"\n==================================================")
        self.log_text.append(f"[*] 啟動任務: {cmd_display}")
        self.log_text.append(f"==================================================")

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.lbl_status.setText(f"執行中: {cmd[1]} ...")

        self.process.setWorkingDirectory(str(Path.cwd()))
        self.process.start(cmd[0], cmd[1:])

    def on_process_output(self):
        data = self.process.readAllStandardOutput()
        text = bytes(data).decode("utf-8", errors="replace")
        clean_text = strip_ansi(text)
        self.log_text.insertPlainText(clean_text)
        # 自動捲動到底部
        cursor = self.log_text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    def on_process_finished(self, exit_code, exit_status):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)

        if exit_code == 0:
            self.lbl_status.setText("✅ 執行成功完成！")
            self.log_text.append(f"\n[+] 任務順利完成 (結束碼: 0)")
            # 若為步驟 2 完成，自動載入 toc.json 內容
            if self.tabs.currentIndex() == 2:
                self.load_toc_json()
        else:
            self.lbl_status.setText(f"❌ 執行中斷或異常 (退出碼: {exit_code})")
            self.log_text.append(f"\n[-] 任務異常結束 (退出碼: {exit_code})")

    def stop_current_process(self):
        if self.process.state() != QProcess.NotRunning:
            self.lbl_status.setText("正在終止進程...")
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
        cmd = [sys.executable, "html_to_epub.py", str(html_p)]
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
        self.execute_cmd_list(cmd)

    def open_folder(self, folder_str: str):
        p = Path(folder_str.strip() or ".")
        if p.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(p.resolve())))
        else:
            QMessageBox.information(self, "提示", f"目錄不存在: {p}")

    def apply_styles(self):
        """套用現代乾淨桌面風格"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f7f9fa;
            }
            QTabWidget::pane {
                border: 1px solid #d0d7de;
                background: white;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #eef2f5;
                border: 1px solid #d0d7de;
                padding: 8px 14px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom-color: white;
                font-weight: bold;
                color: #007acc;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d0d7de;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                border-radius: 4px;
                padding: 4px 10px;
                border: 1px solid #bbb;
                background-color: #f6f8fa;
            }
            QPushButton:hover {
                background-color: #e6ebf1;
            }
            QTableWidget {
                border: 1px solid #d0d7de;
                background-color: white;
                gridline-color: #f0f0f0;
            }
            QHeaderView::section {
                background-color: #f0f3f6;
                padding: 4px;
                border: 1px solid #e1e4e8;
                font-weight: bold;
            }
            QLineEdit, QSpinBox, QComboBox {
                padding: 4px;
                border: 1px solid #ccc;
                border-radius: 3px;
                background: white;
            }
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
    app.setApplicationName("ScanORC GUI")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
