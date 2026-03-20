@echo off
REM pushd maps UNC to a drive letter so paths resolve correctly.
setlocal
pushd "%~dp0.."
for %%I in (.) do git config --global --add safe.directory "%%~fI"
echo Registered safe.directory for %%~fI
popd
endlocal
