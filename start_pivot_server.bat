@echo off
REM PiVot-Server Startup Script (Windows PC)
REM PiVot(Linux) ← → PiVot-Server(Windows) 構成用
REM 使用方法: start_pivot_server.bat

echo 🖥️ Starting PiVot-Server System (Windows PC)
echo Environment: PiVot(Linux) ← → PiVot-Server(Windows)
echo ================================================

REM Windows IP アドレス表示
echo 📍 Windows PC Network Information:
ipconfig | findstr "IPv4"

echo.
echo 🔧 Configuration Steps:
echo 1. Note down your Windows PC IP address above
echo 2. Update config.py on Raspberry Pi with this IP
echo 3. Make sure Windows Firewall allows ports 8001, 8002
echo.

REM ファイアウォール設定の確認
echo 🔒 Checking Windows Firewall...
netsh advfirewall firewall show rule name="PiVot-Server" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Firewall rule not found. Adding rules for PiVot-Server...
    netsh advfirewall firewall add rule name="PiVot-Server-8001" dir=in action=allow protocol=TCP localport=8001
    netsh advfirewall firewall add rule name="PiVot-Server-8002" dir=in action=allow protocol=TCP localport=8002
    echo ✅ Firewall rules added for ports 8001, 8002
) else (
    echo ✅ Firewall rules already exist
)

echo.
echo 🧠 Starting PiVot-Server (NPU Inference)...
start "PiVot-Server NPU" python main_npu.py

REM サーバー起動待機
echo ⏳ Waiting for server startup...
timeout /t 8 /nobreak >nul

REM サーバー状態確認
echo 🔍 Checking server status...
curl -s http://localhost:8001/health >nul 2>&1
if errorlevel 1 (
    echo ❌ Server startup failed or not responding
    echo 💡 Check console window for errors
) else (
    echo ✅ PiVot-Server is running successfully
)

echo.
echo ================================================
echo ✅ PiVot-Server System Ready!
echo.
echo 🔗 Services (Windows PC):
echo    NPU Server: http://localhost:8001
echo    Image Upload: http://localhost:8002
echo    Swagger UI: http://localhost:8001/docs
echo.
echo 🌐 Remote Access (from Raspberry Pi):
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr "IPv4" ^| head -n 1') do set WINDOWS_IP=%%i
echo    NPU Server: http://%WINDOWS_IP:~1%:8001
echo    Image Upload: http://%WINDOWS_IP:~1%:8002
echo.
echo 🎤 Now start PiVot Assistant on Raspberry Pi
echo 🛑 Press any key to stop PiVot-Server
echo ================================================

pause