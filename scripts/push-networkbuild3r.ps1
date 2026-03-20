# Push main + tags to https://github.com/NetworkBuild3r/archivist-oss
# Run from repo root: .\scripts\push-networkbuild3r.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

# .ps1 may be blocked by execution policy; .cmd uses Bypass only for the one-liner inside
& cmd.exe /c "`"$PSScriptRoot\trust-unc-repo.cmd`""

$Url = "https://github.com/NetworkBuild3r/archivist-oss.git"
$has = git remote | Where-Object { $_ -eq "networkbuild3r" }
if (-not $has) {
    git remote add networkbuild3r $Url
} else {
    git remote set-url networkbuild3r $Url
}

git push -u networkbuild3r main --tags
Write-Host "Pushed main and tags to networkbuild3r."
