@echo off
REM Works from UNC: pushd maps the share; git uses -c safe.directory=* so no global config required.
setlocal
pushd "%~dp0.."

git -c safe.directory=* remote set-url origin https://github.com/NetworkBuild3r/archivist-oss.git

git -c safe.directory=* add -A
git -c safe.directory=* diff --cached --quiet
if errorlevel 1 git -c safe.directory=* commit -m "sync: archivist-oss"

git -c safe.directory=* push -u origin main --tags

popd
endlocal
echo Done.
