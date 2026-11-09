@echo off
rem Bambu Colorin: abre el programa y al cerrarlo copia el log a log_colorin.txt (junto a este archivo)
cd /d "%~dp0"
start "" /wait "%~dp0bambu-studio.exe"
set "LOGDIR=%APPDATA%\BambuStudio\log"
for /f "delims=" %%f in ('dir /b /a-d /o-d "%LOGDIR%\*" 2^>nul') do (
  copy /y "%LOGDIR%\%%f" "%~dp0log_colorin.txt" >nul
  goto :hecho
)
echo No se encontro ningun log en %LOGDIR%
pause
exit /b
:hecho
echo Log guardado en "%~dp0log_colorin.txt"
pause
