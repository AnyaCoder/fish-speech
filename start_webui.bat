@echo off

chcp 65001

set no_proxy="localhost, 127.0.0.1, 0.0.0.0"
set PYTHONPATH=%~dp0
set GIT_HOME=%~dp0PortableGit\bin
set PYTHON_FOLDERPATH=%~dp0vits\
set SCRIPT_FOLDERPATH=%~dp0vits\Scripts\

set NEW_PATH="%PYTHON_FOLDERPATH%;%SCRIPT_FOLDERPATH%;%PATH%;"
set PATH=%NEW_PATH%

echo %PATH%
%PYTHON_FOLDERPATH%python fish_speech\webui\manage.py

pause