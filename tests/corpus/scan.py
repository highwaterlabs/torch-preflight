"""Scan a pinned corpus of real training repositories and diff against a baseline.

    python tests/corpus/scan.py                  # scan and diff
    python tests/corpus/scan.py --update-baseline
    python tests/corpus/scan.py --only peft,nanogpt
    python tests/corpus/scan.py --json

Why this exists
---------------
Scanning real training code found **22 bugs in our own rules** across two corpora — see
[spike 0002](../../design/spikes/0002-scanning-real-training-repos.md). Our test fixtures are
written by the same person with the same assumptions as the rules, so they agree with the
rules by construction. Real code disagrees.

Every one of those scans was run by hand with ad-hoc shell, roughly ten times, and the
baselines lived in a scratch directory. This is that, made repeatable.

What a green diff does and does not mean
----------------------------------------
**A corpus diff is evidence about false positives only.** A true positive that stops firing
is indistinguishable from a false positive that got fixed — both appear as a removed line.

This is not hypothetical. A run reporting "zero new findings" once concealed 14 findings
being silenced, 8 of them genuine, and the test suite passed too because every fixture
assigned from something the analysis could resolve. False negatives need fixtures that
*deliberately* exceed what the analysis can see; they cannot be found here.

So the summary always prints what removals mean, rather than letting a clean diff read as a
clean bill of health.

Pinning
-------
Repositories are pinned to a SHA. Without that a diff cannot distinguish "our rules changed"
from "their code changed", which makes every result ambiguous exactly when it matters. Bump a
pin deliberately and re-baseline in the same commit.

``expect_files`` is checked on every run. That guard is not paranoia: a scratch directory was
pruned between two runs, so a comparison scanned **0 files**, reported all 264 findings as
removed, and looked like a spectacular improvement. Scanning nothing must fail loudly.

Not part of the test suite: it needs network and takes minutes. Run it before a release and
after any rule change, like ``tests/calibration/verify_snapshot.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).parent
CORPUS = HERE / "corpus.json"
BASELINE = HERE / "baseline.json"
#: Deliberately **outside** the repository. torch-preflight discovers config by walking up
#: from the scanned path, so a cache under `tests/` inherited this project's own
#: `pyproject.toml` — including `exclude = ["examples"]`, which silently dropped every
#: `examples/` directory in the corpus. `transformers-examples` scanned 0 files and `peft`
#: lost 96 of them, with no error either time.
CACHE = Path(
    os.environ.get("TORCH_PREFLIGHT_CORPUS_CACHE")
    or Path.home() / ".cache" / "torch-preflight-corpus"
)

#: A finding, reduced to what should be stable across runs. The message is included because a
#: reworded message is a real change to what users read, but truncated so that a number
#: inside it does not churn the diff.
Finding = Tuple[str, str, int, str]


def _run(args: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def ensure_checkout(repo: dict) -> Path:
    """Fetch exactly the pinned commit, shallow.

    ``git clone --depth 1`` cannot target a SHA, so this inits and fetches the commit
    directly. GitHub allows that; a host that does not would need a full clone.
    """
    path = CACHE / repo["name"]
    if (path / ".git").exists():
        head = _run(["git", "rev-parse", "HEAD"], cwd=path).stdout.strip()
        if head == repo["sha"]:
            return path
        shutil.rmtree(path)  # pin moved, or a partial checkout was left behind

    path.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "-q"], cwd=path)
    _run(["git", "remote", "add", "origin", repo["url"]], cwd=path)
    if repo.get("sparse"):
        # Written directly rather than via `git sparse-checkout set`, which needs a commit
        # to already exist and silently does nothing in a freshly `git init`-ed repo. That
        # produced a checkout with no files at all, and a scan of zero files.
        _run(["git", "config", "core.sparseCheckout", "true"], cwd=path)
        info = path / ".git" / "info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "sparse-checkout").write_text(f"{repo['sparse']}/*\n")
    fetched = _run(["git", "fetch", "-q", "--depth", "1", "origin", repo["sha"]], cwd=path)
    if fetched.returncode != 0:
        raise SystemExit(
            f"{repo['name']}: could not fetch {repo['sha'][:12]}\n{fetched.stderr.strip()}"
        )
    _run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=path)
    return path


def scan(repo: dict, executable: str) -> Tuple[List[Finding], int]:
    path = ensure_checkout(repo)
    target = path / repo["scan"] if repo.get("scan") else path

    result = _run([executable, "check", str(target), "--format", "json"])
    if not result.stdout.strip():
        raise SystemExit(f"{repo['name']}: torch-preflight produced no output\n{result.stderr}")
    payload = json.loads(result.stdout)

    checked = payload["summary"]["files_checked"]
    expected = repo.get("expect_files")
    if expected is not None and checked != expected:
        raise SystemExit(
            f"{repo['name']}: scanned {checked} files, expected {expected}.\n"
            f"  A scan of the wrong file count makes every finding below meaningless — a\n"
            f"  pruned checkout once reported 0 files and looked like a clean sweep.\n"
            f"  If the pin moved on purpose, update `expect_files` in corpus.json."
        )

    root = str(target)
    findings: List[Finding] = []
    for item in payload["diagnostics"]:
        rel = item["path"]
        rel = rel[len(root):].lstrip("/") if rel.startswith(root) else rel
        findings.append((item["code"], rel, item["line"], item["message"][:60]))
    return sorted(findings), checked


def load_baseline() -> Dict[str, List[Finding]]:
    if not BASELINE.exists():
        return {}
    raw = json.loads(BASELINE.read_text())
    return {name: [tuple(f) for f in rows] for name, rows in raw["findings"].items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--only", help="comma-separated repo names")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--executable", default="torch-preflight")
    args = parser.parse_args()

    repos = json.loads(CORPUS.read_text())["repos"]
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        repos = [r for r in repos if r["name"] in wanted]
        if not repos:
            raise SystemExit(f"no repos matched {args.only!r}")

    baseline = load_baseline()
    current: Dict[str, List[Finding]] = {}
    counts: Dict[str, int] = {}
    for repo in repos:
        print(f"  scanning {repo['name']} ...", file=sys.stderr, flush=True)
        current[repo["name"]], counts[repo["name"]] = scan(repo, args.executable)

    if args.update_baseline:
        merged = {**baseline, **current}
        BASELINE.write_text(json.dumps(
            {"_comment": "Written by scan.py --update-baseline. Re-baseline in the same "
                         "commit as the rule change that moved it.",
             "findings": {k: [list(f) for f in v] for k, v in sorted(merged.items())}},
            indent=1,
        ) + "\n")
        total = sum(len(v) for v in current.values())
        print(f"\nbaseline updated: {len(current)} repos, {total} findings")
        return 0

    added: List[Tuple[str, Finding]] = []
    removed: List[Tuple[str, Finding]] = []
    for name, findings in current.items():
        was = set(baseline.get(name, []))
        now = set(findings)
        added += [(name, f) for f in sorted(now - was)]
        removed += [(name, f) for f in sorted(was - now)]

    if args.json:
        print(json.dumps({
            "added": [[n, *f] for n, f in added],
            "removed": [[n, *f] for n, f in removed],
        }, indent=1))
        return 1 if added else 0

    print(f"\n{'repo':24}{'files':>7}{'findings':>10}")
    for repo in repos:
        name = repo["name"]
        print(f"{name:24}{counts[name]:7}{len(current[name]):10}")
    print(f"{'TOTAL':24}{sum(counts.values()):7}{sum(len(v) for v in current.values()):10}")

    if not baseline:
        print("\nNo baseline yet. Run with --update-baseline to write one.")
        return 0

    if added:
        print(f"\nADDED ({len(added)}) — new findings, most likely new false positives:")
        for name, (code, path, line, message) in added:
            print(f"  {code} {name}/{path}:{line}\n      {message}")
    if removed:
        print(f"\nREMOVED ({len(removed)}):")
        for name, (code, path, line, message) in removed:
            print(f"  {code} {name}/{path}:{line}")

    if not added and not removed:
        print("\nNo change against the baseline.")
    else:
        by_code = Counter(f[0] for _, f in added + removed)
        print(f"\nby rule: {dict(sorted(by_code.items()))}")

    print(
        "\nWhat this run can and cannot tell you:\n"
        "  ADDED findings are evidence of new false positives, and worth reading first.\n"
        "  REMOVED findings are ambiguous. A false positive that got fixed and a true\n"
        "  positive that stopped firing look identical here, so a clean diff is not a\n"
        "  clean bill of health. False negatives need fixtures that deliberately exceed\n"
        "  what the analysis can resolve — see spike 0002 section 5."
    )
    return 1 if added else 0


if __name__ == "__main__":
    raise SystemExit(main())
