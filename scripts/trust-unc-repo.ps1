# Register this clone in Git's safe.directory (fixes "dubious ownership" on UNC \\server\share paths).
# Run once per machine from repo root:
#   scripts\trust-unc-repo.cmd
# If .ps1 is blocked by execution policy:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\trust-unc-repo.ps1
# See: https://git-scm.com/docs/git-config#Documentation/git-config.txt-safedirectory
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$full = (Resolve-Path -LiteralPath $RepoRoot).Path

# Match the form Git itself suggests: %(prefix)///host/share/path
$posix = ($full -replace '^\\\\', '') -replace '\\', '/'
$entry = "%(prefix)///$posix"

$existing = @(git config --global --get-all safe.directory 2>$null)
if ($existing -contains $entry) {
    Write-Host "safe.directory already set for this repo."
    exit 0
}

git config --global --add safe.directory $entry
Write-Host "Registered: $entry"
Write-Host "You can run git status / commit / push now."
