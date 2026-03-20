# Backward-compatible wrapper: pushes to GitHub using the same remote name as before.
# Prefer: .\scripts\publish-remotes.ps1 -PublicUrl "git@github.com:NetworkBuild3r/archivist-oss.git" -PublicRemoteName origin -WithTags
# If your repo uses `origin` for GitHub, pass -PublicRemoteName origin (default in this script).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
& "$Root\scripts\publish-remotes.ps1" `
    -PublicUrl "git@github.com:NetworkBuild3r/archivist-oss.git" `
    -PublicRemoteName "origin" `
    -WithTags `
    @args
