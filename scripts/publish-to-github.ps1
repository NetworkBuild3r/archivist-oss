# Run from repo root (archivist-oss): .\scripts\publish-to-github.ps1
$ErrorActionPreference = "Stop"
$Remote = "git@github.com:AHEAD-Labs/ai-archivist-oss.git"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path ".git")) {
    git init
}
git add -A
$status = git status --porcelain
if ($status) {
    git commit -m "Archivist OSS v0.3.0: multi-agent fleet memory, RLM pipeline"
}
git branch -M main
$remotes = git remote
if ($remotes -match "origin") {
    git remote set-url origin $Remote
} else {
    git remote add origin $Remote
}
git push -u origin main --tags
Write-Host "Done. Remote: $Remote"
