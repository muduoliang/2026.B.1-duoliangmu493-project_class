@echo off
REM ===========================================================================
REM  run.bat —— 一行启动本地网页应用（Windows）
REM  用法：双击本文件，或在本目录执行 run.bat
REM  可选：
REM    set RL_WEB_PORT=8010   &  run.bat       指定端口（默认 8000）
REM    set RL_WEB_PYTHON=py路径 & run.bat      指定 Python 解释器
REM  说明：需要 numpy + torch 的解释器；优先使用下面 detect 到的 conda 环境，
REM        找不到时回退到 PATH 里的 python。
REM ===========================================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "PY="
if defined RL_WEB_PYTHON set "PY=%RL_WEB_PYTHON%"
if not defined PY if exist "E:\project\2025.A.01\anaconde\envs\cityflow\python.exe" set "PY=E:\project\2025.A.01\anaconde\envs\cityflow\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\envs\cityflow\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\cityflow\python.exe"
if not defined PY set "PY=python"

echo [run] 解释器: %PY%
"%PY%" -c "import sys;print('[run] Python',sys.version.split()[0])" || goto :err
"%PY%" -c "import torch" >nul 2>nul || echo [run] 警告：该解释器缺少 torch，RL 打分将不可用
"%PY%" app.py %*
goto :eof

:err
echo.
echo [run] 无法启动：请安装 numpy + torch，或用 set RL_WEB_PYTHON=... 指定解释器
pause
