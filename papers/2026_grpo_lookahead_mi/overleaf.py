"""Two-way sync between this paper folder and its Overleaf project (Overleaf Git access).

    & ..\\..\\.venv\\Scripts\\python.exe overleaf.py init https://git.overleaf.com/<project-id>
    & ..\\..\\.venv\\Scripts\\python.exe overleaf.py status   # what differs, in both directions
    & ..\\..\\.venv\\Scripts\\python.exe overleaf.py pull     # Overleaf -> this folder
    & ..\\..\\.venv\\Scripts\\python.exe overleaf.py push     # this folder -> Overleaf

The Overleaf project holds ONLY the files that compile the paper -- the same list
``make_overleaf_zip.py`` uses, imported here so there is one definition of "what Overleaf needs".
The ledger, the READMEs, the render/build scripts and the review notes stay in this repo and are
never pushed.

Rather than wire Overleaf into this repo's history (the paper is a subdirectory, so that would
mean ``git subtree``, which entangles two histories and would also expose NUMBERS.md and the
review notes), this keeps a private clone of the Overleaf repo OUTSIDE the tree -- by default
``%LOCALAPPDATA%\\overleaf-mirrors\\<folder name>``, overridable with $OVERLEAF_MIRROR -- and
copies the managed files across. The thesis repo's history stays linear and Overleaf-free.

Both directions overwrite whole files, so ``push`` refuses when the Overleaf project has moved
since the last sync (run ``pull`` first, or ``push --force`` to deliberately discard the Overleaf
side), and ``pull`` refuses when the paper folder has uncommitted changes that it would clobber.
``status`` never touches the paper folder (it does refresh the throwaway clone, so that what it
reports is the state of the Overleaf project right now).

Overleaf authentication: generate a Git token at Overleaf → Account Settings → Git integration,
and give it as the PASSWORD at the git prompt (the username is ignored). Windows' credential
manager stores it after the first time.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_overleaf_zip import FILES as SOURCE_FILES  # noqa: E402  (single source of truth)

_default_mirror = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "overleaf-mirrors" / HERE.name
MIRROR = Path(os.environ.get("OVERLEAF_MIRROR") or _default_mirror)
STATE = MIRROR / ".git" / "overleaf-last-synced"

# Files this tool owns on the Overleaf side: anything matching these is pushed, and anything
# matching them that is no longer a source file is deleted there (so a renamed section does not
# linger and keep compiling).
MANAGED = ("main.tex", "refs.bib", "acl.sty", "acl_natbib.bst", "sections/*.tex", "figures/*.png")


def git(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd or MIRROR, capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r.stdout.strip()


def need_mirror() -> None:
    if not (MIRROR / ".git").exists():
        raise SystemExit(f"no Overleaf clone at {MIRROR}\nRun:  overleaf.py init <git-url>")


def branch() -> str:
    return git("rev-parse", "--abbrev-ref", "HEAD")


TEXT_SUFFIXES = {".tex", ".bib", ".sty", ".bst"}


def same(a: Path, b: Path) -> bool:
    """Same content? For text files, ignoring line endings.

    Windows checks LaTeX sources out as CRLF while git and Overleaf both store LF, so a byte
    comparison marks files identical in every way that matters as different -- and copying them
    over would be churn, not a change. LaTeX does not care either way. Figures compare as bytes.
    """
    if a.suffix.lower() not in TEXT_SUFFIXES:
        return filecmp.cmp(a, b, shallow=False)
    norm = lambda p: p.read_bytes().replace(b"\r\n", b"\n")  # noqa: E731
    return norm(a) == norm(b)


def managed_in(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for pattern in MANAGED:
        for p in sorted(root.glob(pattern)):
            if p.is_file():
                out[p.relative_to(root).as_posix()] = p
    return out


def sources() -> dict[str, Path]:
    return {p.relative_to(HERE).as_posix(): p for p in SOURCE_FILES}


def compare() -> tuple[list[str], list[str], list[str]]:
    """(differing, only-here, only-on-overleaf) over the managed files."""
    here, there = sources(), managed_in(MIRROR)
    differing = [k for k in sorted(here.keys() & there.keys()) if not same(here[k], there[k])]
    return differing, sorted(here.keys() - there.keys()), sorted(there.keys() - here.keys())


def remote_moved() -> bool:
    git("fetch", "origin", "--quiet")
    remote = git("rev-parse", f"origin/{branch()}")
    return not STATE.exists() or STATE.read_text().strip() != remote


def mark_synced() -> None:
    STATE.write_text(git("rev-parse", f"origin/{branch()}") + "\n")


def incoming() -> list[str]:
    """Overleaf commits made since the last sync: who edited, when, what they called it.

    Overleaf squashes web editing into commits authored by whoever made them, so this is how a
    supervisor's edits announce themselves. Their *comments* are NOT here -- Overleaf keeps
    review-panel comments outside the file content, so git never sees them.
    """
    if not STATE.exists():
        return []
    old = STATE.read_text().strip()
    log = git("log", "--format=%h  %an  %ar  %s", f"{old}..origin/{branch()}", check=False)
    return [l for l in log.splitlines() if l.strip()]


def repo_dirty() -> list[str]:
    out = git("status", "--porcelain", "--", str(HERE), cwd=HERE)
    return [l for l in out.splitlines() if l.strip()]


def cmd_init(url: str) -> int:
    if (MIRROR / ".git").exists():
        raise SystemExit(f"already initialised at {MIRROR}\nDelete that folder to re-clone.")
    MIRROR.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning {url}\n     -> {MIRROR}")
    # -c core.autocrlf=false: git's Windows default rewrites LF to CRLF on checkout, which would
    # make every text file in the clone differ from its LF original here on byte comparison --
    # 18 files "changed" with identical content. Compare and copy raw bytes, both ways.
    git("clone", "-c", "core.autocrlf=false", "-c", "core.eol=lf", url, str(MIRROR),
        cwd=MIRROR.parent)
    mark_synced()
    print(f"ok. Overleaf branch '{branch()}'. Next:  overleaf.py status")
    return 0


def cmd_status() -> int:
    need_mirror()
    moved = remote_moved()                      # fetches
    # Compare against what Overleaf actually HAS, not the clone's last-seen working tree, or a
    # project that has moved reports "differ: 0". The clone is disposable scratch, never edited
    # by hand, so resetting it here changes nothing anyone owns; the paper folder is untouched.
    git("reset", "--hard", f"origin/{branch()}", "--quiet")
    differing, only_here, only_there = compare()
    print(f"Overleaf clone: {MIRROR}  (branch {branch()})")
    print(f"Overleaf project changed since last sync: {'YES -- pull first' if moved else 'no'}")
    for line in incoming():
        print("     ", line)
    for label, items in (("differ", differing), ("only in this repo", only_here),
                         ("only on Overleaf", only_there)):
        print(f"  {label}: {len(items)}")
        for i in items:
            print("     ", i)
    dirty = repo_dirty()
    if dirty:
        print(f"  uncommitted in this folder: {len(dirty)} (commit before pulling)")
    if not (differing or only_here or only_there):
        print("  in sync")
    return 0


def cmd_pull() -> int:
    need_mirror()
    if repo_dirty():
        raise SystemExit("this folder has uncommitted changes; commit them first "
                         "(pull overwrites files from Overleaf).\n"
                         + "\n".join("  " + l for l in repo_dirty()))
    git("fetch", "origin", "--quiet")
    for line in incoming():
        print("  Overleaf commit:", line)
    git("reset", "--hard", f"origin/{branch()}", "--quiet")
    changed = []
    for rel, src in managed_in(MIRROR).items():
        dst = HERE / rel
        if not dst.exists() or not same(src, dst):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            changed.append(rel)
    mark_synced()
    print(f"pulled from Overleaf: {len(changed)} file(s) updated")
    for c in changed:
        print("  ", c)
    if changed:
        print("Review with `git diff`, rebuild with build.py, then commit.")
    return 0


def cmd_push(message: str, force: bool) -> int:
    need_mirror()
    if remote_moved() and not force:
        raise SystemExit("the Overleaf project has changed since the last sync.\n"
                         "Run `overleaf.py pull` first (or `push --force` to discard "
                         "the Overleaf-side changes).")
    git("fetch", "origin", "--quiet")
    git("reset", "--hard", f"origin/{branch()}", "--quiet")
    here = sources()
    changed = []
    for rel, src in here.items():
        dst = MIRROR / rel
        if not dst.exists() or not same(src, dst):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            changed.append(rel)
    for rel, path in managed_in(MIRROR).items():        # propagate deletions/renames
        if rel not in here:
            path.unlink()
            changed.append(f"{rel} (deleted)")
    if not changed:
        print("Overleaf is already up to date")
        return 0
    git("add", "-A")
    if not git("status", "--porcelain"):        # e.g. differences git normalises away
        print("Overleaf is already up to date (no net change)")
        return 0
    git("commit", "-m", message)
    git("push", "origin", branch(), "--quiet")
    mark_synced()
    print(f"pushed to Overleaf: {len(changed)} file(s)")
    for c in changed:
        print("  ", c)
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_init = sub.add_parser("init", help="clone the Overleaf project (one time)")
    p_init.add_argument("url", help="https://git.overleaf.com/<project-id>")
    sub.add_parser("status", help="show what differs, change nothing")
    sub.add_parser("pull", help="Overleaf -> this folder")
    p_push = sub.add_parser("push", help="this folder -> Overleaf")
    p_push.add_argument("-m", "--message", default="Update from Claude Code")
    p_push.add_argument("--force", action="store_true",
                        help="push even though Overleaf changed (discards those changes)")
    a = ap.parse_args(argv)
    if a.cmd == "init":
        return cmd_init(a.url)
    if a.cmd == "status":
        return cmd_status()
    if a.cmd == "pull":
        return cmd_pull()
    return cmd_push(a.message, a.force)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
