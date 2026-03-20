# Internal + public Git remotes

## UNC / “dubious ownership”

If the repo lives on a network path (`\\192.168…\…`) and Git refuses commands, run **once** from the repo root:

```bat
scripts\trust-unc-repo.cmd
```

If PowerShell blocks unsigned scripts, **do not** rely on `.\trust-unc-repo.ps1` alone — use the `.cmd` above, or:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\trust-unc-repo.ps1
```

This adds the path Git expects to `git config --global safe.directory` (see Git’s `%(prefix)///…` form for UNC).

---

Use **one local clone** with **two push targets**:

| Remote    | Typical URL                         | Purpose |
|-----------|-------------------------------------|---------|
| `internal`| Private GitLab / GitHub Enterprise  | Day-to-day work, team PRs, CI, secrets allowed in *private* CI only — never commit secrets. |
| `public`  | Public GitHub (`…/ai-archivist-oss`) | Open-source mirror: same `main` + tags, no internal hostnames or credentials in tree. |

## First-time setup

From the repo root (`archivist-oss`):

```powershell
# Internal (replace with your team URL)
git remote add internal https://gitlab.example.com/your-org/archivist-oss.git

# Public GitHub — many clones already use the name `origin` for this:
#   git remote -v
# If origin is GitHub, you do NOT need a second name; use scripts with -PublicRemoteName origin
```

If `origin` already points at GitHub, add **only** the internal remote:

```powershell
git remote add internal https://your-gitlab.example.com/org/archivist-oss.git
```

Optional: rename for clarity (`origin` → public name):

```powershell
git remote rename origin public
```

## Daily workflow

1. Commit and push to **`internal` first** (team sees changes immediately).

   ```powershell
   git push internal main
   ```

2. When a release is ready for the world, run the publish script (pushes to `public` and optionally `internal`), or:

   ```powershell
   git push public main --tags
   ```

## Before every public push

- [ ] No real URLs to internal clusters, VPN, or staging in committed files (use `.env.example` only).
- [ ] No `team_map.yaml` / `namespaces.yaml` with real agent names if those are sensitive — ship `.example` files only.
- [ ] `git grep -iE 'password|secret|token|api_key'` on changed files.

## Automation

See [`scripts/publish-remotes.ps1`](../scripts/publish-remotes.ps1) for parameterized pushes to both remotes.
