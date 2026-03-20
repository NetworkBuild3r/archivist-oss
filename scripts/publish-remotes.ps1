<#
.SYNOPSIS
  Push main (+ optional tags) to internal and/or public remotes.

.DESCRIPTION
  Configures remotes if missing, then pushes. Use for team-private + public OSS mirror.

.EXAMPLE
  .\scripts\publish-remotes.ps1 -InternalUrl "https://gitlab.example.com/org/archivist.git" -PublicUrl "git@github.com:NetworkBuild3r/archivist-oss.git" -WithTags

.EXAMPLE
  .\scripts\publish-remotes.ps1 -PublicUrl "git@github.com:NetworkBuild3r/archivist-oss.git" -WithTags
#>
param(
    [string] $InternalUrl = "",
    [string] $PublicUrl = "git@github.com:NetworkBuild3r/archivist-oss.git",
    [string] $InternalRemoteName = "internal",
    [string] $PublicRemoteName = "public",
    [switch] $WithTags,
    [switch] $SkipCommit
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

& cmd.exe /c "`"$PSScriptRoot\trust-unc-repo.cmd`""

if (-not (Test-Path ".git")) {
    Write-Error "Not a git repository. Run from archivist-oss root."
}

function Ensure-Remote {
    param([string]$Name, [string]$Url)
    if ([string]::IsNullOrWhiteSpace($Url)) { return }
    $exists = git remote | Where-Object { $_ -eq $Name }
    if ($exists) {
        git remote set-url $Name $Url
        Write-Host "Updated remote $Name -> $Url"
    } else {
        git remote add $Name $Url
        Write-Host "Added remote $Name -> $Url"
    }
}

if (-not $SkipCommit) {
    git add -A
    $status = git status --porcelain
    if ($status) {
        git commit -m "chore: sync before push"
    }
}

git branch -M main 2>$null

if (-not [string]::IsNullOrWhiteSpace($PublicUrl)) {
    Ensure-Remote -Name $PublicRemoteName -Url $PublicUrl
}
if (-not [string]::IsNullOrWhiteSpace($InternalUrl)) {
    Ensure-Remote -Name $InternalRemoteName -Url $InternalUrl
}

if (-not [string]::IsNullOrWhiteSpace($InternalUrl)) {
    Write-Host "Pushing to ${InternalRemoteName}..."
    if ($WithTags) {
        git push $InternalRemoteName main --tags
    } else {
        git push $InternalRemoteName main
    }
}

if (-not [string]::IsNullOrWhiteSpace($PublicUrl)) {
    Write-Host "Pushing to ${PublicRemoteName}..."
    if ($WithTags) {
        git push $PublicRemoteName main --tags
    } else {
        git push $PublicRemoteName main
    }
}

Write-Host "Done."
