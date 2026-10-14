#!/usr/bin/env bash
# ScanORC Debian Package Build Script (Plan A: Thin Architecture)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PKG_NAME="scanorc"
DEFAULT_VER="1.0.0-1~mint22.3"
VERSION="${1:-$DEFAULT_VER}"
ARCH="amd64"
DEB_NAME="${PKG_NAME}_${VERSION}_${ARCH}.deb"

BUILD_ROOT="$PROJECT_ROOT/deb_dist"
STAGING_DIR="$BUILD_ROOT/staging"

echo "============================================================"
echo "  開始建置 ScanORC Debian 安裝包 (.deb)"
echo "  套件名稱: $PKG_NAME"
echo "  套件版本: $VERSION"
echo "  目標架構: $ARCH"
echo "============================================================"

# 清理舊有暫存
rm -rf "$BUILD_ROOT"
mkdir -p "$STAGING_DIR/DEBIAN"
mkdir -p "$STAGING_DIR/opt/scanorc"
mkdir -p "$STAGING_DIR/usr/bin"
mkdir -p "$STAGING_DIR/usr/share/applications"
mkdir -p "$STAGING_DIR/usr/share/icons/hicolor/scalable/apps"
mkdir -p "$STAGING_DIR/usr/share/icons/hicolor/256x256/apps"
mkdir -p "$STAGING_DIR/usr/share/icons/hicolor/128x128/apps"
mkdir -p "$STAGING_DIR/usr/share/icons/hicolor/64x64/apps"
mkdir -p "$STAGING_DIR/usr/share/icons/hicolor/48x48/apps"

# 1. 複製核心 Python 腳本與相依檔案至 /opt/scanorc/
echo "[1/6] 複製 ScanORC 核心程式碼至 /opt/scanorc/ ..."
cd "$PROJECT_ROOT"

# 複製 Python 程式碼 (不含 models, samples, venv, git, 測試暫存)
cp -a app_gui.py "$STAGING_DIR/opt/scanorc/"
cp -a process_book.py "$STAGING_DIR/opt/scanorc/"
cp -a pipeline.py "$STAGING_DIR/opt/scanorc/"
cp -a extract_toc.py "$STAGING_DIR/opt/scanorc/"
cp -a preprocess_scantailor.py "$STAGING_DIR/opt/scanorc/"
cp -a batch_process.py "$STAGING_DIR/opt/scanorc/"
cp -a patch_pages.py "$STAGING_DIR/opt/scanorc/"
cp -a html_to_epub.py "$STAGING_DIR/opt/scanorc/"
cp -a recompress_images.py "$STAGING_DIR/opt/scanorc/"
cp -a download_models.py "$STAGING_DIR/opt/scanorc/"
cp -a config_utils.py "$STAGING_DIR/opt/scanorc/"
cp -a orientation_detector.py "$STAGING_DIR/opt/scanorc/"
cp -a requirements.txt "$STAGING_DIR/opt/scanorc/"
cp -a LICENSE "$STAGING_DIR/opt/scanorc/"

# 保持軟連結
(cd "$STAGING_DIR/opt/scanorc" && ln -sf recompress_images.py recompress_jpg.py)
(cd "$STAGING_DIR/opt/scanorc" && ln -sf recompress_images.py images_compressed.py)

# 複製 bin/ (portable scantailor-cli 工具鏈)
echo "[2/6] 複製免安裝 ScanTailor 工具鏈 (bin/) ..."
cp -a bin "$STAGING_DIR/opt/scanorc/"

# 2. 複製執行檔 Wrappers 至 /usr/bin/
echo "[3/6] 安裝系統指令 Wrappers 至 /usr/bin/ ..."
cp -a "$SCRIPT_DIR/wrappers/scanorc" "$STAGING_DIR/usr/bin/"
cp -a "$SCRIPT_DIR/wrappers/scanorc-gui" "$STAGING_DIR/usr/bin/"
cp -a "$SCRIPT_DIR/wrappers/scanorc-models" "$STAGING_DIR/usr/bin/"
cp -a "$SCRIPT_DIR/wrappers/scanorc-batch" "$STAGING_DIR/usr/bin/"
cp -a "$SCRIPT_DIR/wrappers/scanorc-toc" "$STAGING_DIR/usr/bin/"
cp -a "$SCRIPT_DIR/wrappers/scanorc-orientation" "$STAGING_DIR/usr/bin/"
cp -a "$SCRIPT_DIR/wrappers/scanorc-pipeline" "$STAGING_DIR/usr/bin/"
cp -a "$SCRIPT_DIR/wrappers/scanorc-setup" "$STAGING_DIR/usr/bin/"

# 3. 複製桌面整合檔案 (.desktop 與圖示)
echo "[4/6] 產生桌面捷徑與多解析度圖示..."
cp -a "$SCRIPT_DIR/scanorc.desktop" "$STAGING_DIR/usr/share/applications/"
cp -a "$SCRIPT_DIR/scanorc.svg" "$STAGING_DIR/usr/share/icons/hicolor/scalable/apps/"

if command -v inkscape >/dev/null 2>&1; then
    inkscape "$SCRIPT_DIR/scanorc.svg" -w 256 -h 256 -o "$STAGING_DIR/usr/share/icons/hicolor/256x256/apps/scanorc.png" 2>/dev/null || cp "$SCRIPT_DIR/scanorc.png" "$STAGING_DIR/usr/share/icons/hicolor/256x256/apps/"
    inkscape "$SCRIPT_DIR/scanorc.svg" -w 128 -h 128 -o "$STAGING_DIR/usr/share/icons/hicolor/128x128/apps/scanorc.png" 2>/dev/null || true
    inkscape "$SCRIPT_DIR/scanorc.svg" -w 64 -h 64 -o "$STAGING_DIR/usr/share/icons/hicolor/64x64/apps/scanorc.png" 2>/dev/null || true
    inkscape "$SCRIPT_DIR/scanorc.svg" -w 48 -h 48 -o "$STAGING_DIR/usr/share/icons/hicolor/48x48/apps/scanorc.png" 2>/dev/null || true
elif [ -f "$SCRIPT_DIR/scanorc.png" ]; then
    cp -a "$SCRIPT_DIR/scanorc.png" "$STAGING_DIR/usr/share/icons/hicolor/256x256/apps/"
fi

# 4. 配置 DEBIAN 元資料與維護者腳本
echo "[5/6] 產生 DEBIAN/control 與維護者腳本..."
# 計算 Installed-Size (以 KB 為單位)
INSTALLED_SIZE=$(du -sk "$STAGING_DIR" | cut -f1)

sed -e "s/^Version:.*/Version: $VERSION/" "$SCRIPT_DIR/control" > "$STAGING_DIR/DEBIAN/control"
echo "Installed-Size: $INSTALLED_SIZE" >> "$STAGING_DIR/DEBIAN/control"

cp -a "$SCRIPT_DIR/postinst" "$STAGING_DIR/DEBIAN/"
cp -a "$SCRIPT_DIR/prerm" "$STAGING_DIR/DEBIAN/"
cp -a "$SCRIPT_DIR/postrm" "$STAGING_DIR/DEBIAN/"

mkdir -p "$STAGING_DIR/usr/share/doc/scanorc"
cp -a "$PROJECT_ROOT/LICENSE" "$STAGING_DIR/usr/share/doc/scanorc/copyright"

# 5. 正規化權限
find "$STAGING_DIR" -type d -exec chmod 755 {} +
find "$STAGING_DIR" -type f -exec chmod 644 {} +
chmod 755 "$STAGING_DIR/DEBIAN/postinst" "$STAGING_DIR/DEBIAN/prerm" "$STAGING_DIR/DEBIAN/postrm"
chmod 755 "$STAGING_DIR/usr/bin"/*
chmod 755 "$STAGING_DIR/opt/scanorc"/*.py
chmod 755 "$STAGING_DIR/opt/scanorc/bin/scantailor-cli"
chmod 755 "$STAGING_DIR/opt/scanorc/bin/scantailor-cli-bin"
find "$STAGING_DIR/opt/scanorc/bin/lib" -type f -name "*.so*" -exec chmod 644 {} + 2>/dev/null || true

# 6. 建置 .deb 包
echo "[6/6] 呼叫 dpkg-deb 打包..."
FINAL_DEB="$BUILD_ROOT/$DEB_NAME"
dpkg-deb --build --root-owner-group "$STAGING_DIR" "$FINAL_DEB"

echo "============================================================"
echo "[🎉] Debian 安裝包建置成功！"
echo "檔案位置: $FINAL_DEB"
ls -lh "$FINAL_DEB"
echo "============================================================"
echo "[*] 套件內容概覽："
dpkg -c "$FINAL_DEB" > "$BUILD_ROOT/deb_files.txt"
head -n 25 "$BUILD_ROOT/deb_files.txt"
echo "..."
tail -n 15 "$BUILD_ROOT/deb_files.txt"
echo "============================================================"
