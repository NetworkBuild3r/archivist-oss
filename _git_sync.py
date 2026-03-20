"""One-shot: commit all + push origin (public). Run: python _git_sync.py"""
import subprocess
import sys

REPO = r"\\192.168.11.102\k8s-argocd\openclaw\agents\nova\archivist-oss"
G = ["git", "-c", "safe.directory=*", "-C", REPO]


def run(args, **kw):
    r = subprocess.run(G + args, capture_output=True, text=True, **kw)
    open(REPO + "\\_git_sync_log.txt", "a", encoding="utf-8").write(
        f"\n$ {' '.join(G + args)}\nexit={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}\n"
    )
    return r


def main():
    open(REPO + "\\_git_sync_log.txt", "w", encoding="utf-8").write("start\n")
    run(["remote", "set-url", "origin", "https://github.com/NetworkBuild3r/archivist-oss.git"])
    run(["add", "-A"])
    st = run(["status", "--porcelain"])
    if st.stdout.strip():
        run(["commit", "-m", "sync: archivist-oss public mirror"])
    r = run(["push", "-u", "origin", "main", "--tags"])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
