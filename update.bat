@echo off
chcp 65001

%~dp0PortableGit\bin\git pull
echo "更新完毕"
pause