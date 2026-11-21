@echo off
echo Building tse_option_v2.exe ...
python -m PyInstaller --noconfirm --onefile --windowed --name tse_option_v2 app.py
if exist dist\tse_option_v2.exe (
    copy /Y config.json dist\config.json >nul
    echo.
    echo ==========================================
    echo Build succeeded! Check dist\tse_option_v2.exe
    echo ==========================================
) else (
    echo Build failed!
)
pause
