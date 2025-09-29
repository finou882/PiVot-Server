# PiVot Installation Troubleshooting Guide

このガイドでは、PiVotセットアップ時によくある問題と解決方法を説明します。

## 🔧 一般的な問題と解決方法

### 1. PyAudio インストールエラー

**エラー例:**
```
error: Microsoft Visual C++ 14.0 is required
fatal error: 'portaudio.h' file not found
ImportError: /home/pi/miniconda3/lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found
Could not import the PyAudio C module 'pyaudio._portaudio'
```

**解決方法:**

#### Conda環境での競合問題:
```bash
# 問題診断
python3 check_audio.py

# 自動修復を試す
bash fix_pyaudio.sh

# または、システムPythonを使用
conda deactivate
/usr/bin/python3 main.py

# システムPyAudioをインストール
sudo apt install -y python3-pyaudio portaudio19-dev
```

#### Raspberry Pi / Linux:
```bash
# システム依存関係をインストール
sudo apt update
sudo apt install -y portaudio19-dev python3-pyaudio

# それでもダメな場合
sudo apt install -y libasound2-dev libportaudio2 libportaudiocpp0
pip3 install --upgrade pip setuptools wheel
pip3 install pyaudio --no-cache-dir
```

#### Windows:
```powershell
# 事前ビルドされたホイールを使用
pip install pipwin
pipwin install pyaudio

# または
pip install pyaudio --only-binary=all
```

### 2. PiCamera インストールエラー

**エラー例:**
```
Could not find a Raspberry Pi camera module
error: can't find Rust compiler
```

**解決方法:**

#### Raspberry Pi:
```bash
# システムパッケージを使用
sudo apt install -y python3-picamera

# カメラを有効化
sudo raspi-config
# Interface Options > Camera > Enable を選択

# システム再起動
sudo reboot
```

#### カメラなしでテスト:
```bash
# カメラ機能を無効にしてテスト
export DISABLE_CAMERA=1
python3 main.py
```

### 3. Intel NPU ドライバーエラー

**エラー例:**
```
No Intel NPU device found
Intel AI Boost driver not detected
```

**解決方法:**

#### Windows (Intel Meteor Lake/Lunar Lake):
1. Intel AI Boost NPU ドライバーをインストール:
   - [Intel ドライバー & サポート](https://www.intel.com/content/www/us/en/support/products/230857/processors/intel-core-processors.html)
2. デバイスマネージャーで確認:
   - "システム デバイス" → "Intel(R) AI Boost"
3. 再起動後に再テスト

#### NPU なしでの動作:
```python
# CPU フォールバックを使用
export USE_CPU_FALLBACK=1
python3 main.py
```

### 4. ネットワーク接続エラー

**エラー例:**
```
Connection refused to Windows PC
Could not detect Windows PC IP
ARP table scan failed: [Errno 2] No such file or directory: 'arp'
DeprecationWarning: There is no current event loop
```

**解決方法:**

#### Windows PC側:
```powershell
# ファイアウォール設定を確認
netsh advfirewall firewall add rule name="PiVot Server" dir=in action=allow port=8000 protocol=TCP

# PiVot-Serverを起動
python main.py
```

#### Raspberry Pi側:
```bash
# 更新されたnetwork_setup.pyを使用
python3 network_setup.py

# IP アドレスを手動設定
nano config.py
# WINDOWS_PC_IP = "192.168.68.xxx" に変更

# ネットワークテスト
python3 test_connection.py

# ARPコマンド不足の場合
sudo apt install net-tools

# または直接接続テスト
ping 192.168.68.xxx  # Windows PCのIP
curl http://192.168.68.xxx:8000/health
```

### 5. 権限エラー

**エラー例:**
```
Permission denied
sudo: command not found
```

**解決方法:**

#### Linux/Raspberry Pi:
```bash
# sudoを使用してインストール
sudo bash setup_all.sh

# ユーザーをaudioグループに追加
sudo usermod -a -G audio $USER

# 再ログインして権限を反映
```

### 6. Python バージョンエラー

**エラー例:**
```
Python 3.8+ is required
SyntaxError: invalid syntax
```

**解決方法:**

#### バージョン確認:
```bash
python3 --version
# Python 3.8+ が必要
```

#### アップデート (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install -y python3.9 python3.9-pip
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
```

## 🚀 代替インストール方法

### 方法1: システムパッケージを優先
```bash
# Raspberry Pi
sudo apt install -y python3-pyaudio python3-picamera python3-opencv python3-numpy
pip3 install fastapi uvicorn requests

# Ubuntu/Debian
sudo apt install -y python3-pyaudio python3-opencv python3-numpy
pip3 install fastapi uvicorn requests picamera2
```

### 方法2: conda環境を使用
```bash
# condaをインストール
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh

# 環境作成
conda create -n pivot python=3.9
conda activate pivot
conda install numpy opencv
pip install -r requirements.txt
```

### 方法3: Docker使用
```bash
# Dockerでコンテナ実行
docker build -t pivot .
docker run -p 8000:8000 --device=/dev/video0 pivot
```

## 📋 診断コマンド

### システム情報確認:
```bash
# OS情報
cat /etc/os-release

# Python バージョン
python3 --version

# インストール済みパッケージ
pip3 list | grep -E "(pyaudio|opencv|numpy|fastapi)"

# カメラデバイス確認
ls /dev/video*
```

### ネットワーク診断:
```bash
# IP アドレス確認
hostname -I

# Windows PC への接続テスト
ping 192.168.1.100

# ポート確認
nmap -p 8000 192.168.1.100
```

### NPU デバイス確認:
```bash
# Linux でのデバイス確認
lspci | grep -i intel
lsusb | grep -i intel

# Windows でのデバイス確認 (PowerShell)
Get-WmiObject Win32_PnPEntity | Where-Object {$_.Name -like "*Intel*AI*"}
```

## 🆘 サポート

問題が解決しない場合は、以下の情報を含めてサポートに連絡してください:

1. **環境情報:**
   - OS: (Raspberry Pi OS, Ubuntu, Windows)
   - Python バージョン
   - ハードウェア情報

2. **エラーメッセージ:**
   - 完全なエラーログ
   - コマンド実行履歴

3. **診断結果:**
   ```bash
   # 診断情報収集
   python3 --version > debug_info.txt
   pip3 list >> debug_info.txt
   lsb_release -a >> debug_info.txt
   ```

## 🔄 クリーンインストール

完全にリセットして最初からやり直す場合:

```bash
# 仮想環境削除
rm -rf venv

# キャッシュクリア
pip3 cache purge
rm -rf ~/.cache/pip

# 設定ファイル削除
rm -f config.env

# 新規インストール
curl -sSL https://raw.githubusercontent.com/your-repo/pivot-sever/main/PiVot/setup_all.sh | bash
```

---

このトラブルシューティングガイドで問題が解決しない場合は、GitHubのIssuesページで報告してください。