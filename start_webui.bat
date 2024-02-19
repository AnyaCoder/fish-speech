@echo off

chcp 65001

set no_proxy="localhost, 127.0.0.1, 0.0.0.0"
set PYTHONPATH=%~dp0
set FFMPEG=%~dp0ffmpeg\bin\
set PYTHON_FOLDERPATH=D:\miniconda\envs\vits\
set SCRIPT_FOLDERPATH=D:\miniconda\envs\vits\Scripts\
set CARGO=%USERPROFILE%\.cargo\bin\
set RUST_LOG=info

set NEW_PATH="%FFMPEG%;%CARGO%;%PYTHON_FOLDERPATH%;%SCRIPT_FOLDERPATH%;%PATH%"
set PATH=%NEW_PATH%

echo %PATH%
%PYTHON_FOLDERPATH%python fish_speech\webui\manage.py
pause