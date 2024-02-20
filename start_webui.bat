@echo off

chcp 65001

set no_proxy="localhost, 127.0.0.1, 0.0.0.0"
set PYTHONPATH=%~dp0
set GIT_HOME=%~dp0PortableGit\bin
set PYTHON_FOLDERPATH=D:\miniconda\envs\vits\
set SCRIPT_FOLDERPATH=D:\miniconda\envs\vits\Scripts\
set CARGO=%USERPROFILE%\.cargo\bin\

set NEW_PATH="%CARGO%;%PYTHON_FOLDERPATH%;%SCRIPT_FOLDERPATH%;%PATH%;"
set PATH=%NEW_PATH%

echo %PATH%
%PYTHON_FOLDERPATH%python fish_speech\webui\manage.py

pause