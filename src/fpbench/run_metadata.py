"""Run metadata: what produced a results file.

A CSV in `results/data/` records the axes a sweep varied (bits, seed, format,
target). It records nothing about the things held fixed, and those live only as
module constants that a command-line flag can silently override. `--epochs 3`
produces a file indistinguishable from the 12-epoch protocol.

`record_run(...)` writes a sibling `<name>.meta.json` holding the rest: the
commit the code was at, whether the working tree was dirty, the exact argv, the
resolved configuration including defaults, and the machine. Reading a CSV and
its metadata file together is enough to say what protocol produced it.

    with record_run(out, config=CONFIG, args=args) as meta:
        ...                       # the sweep
        meta["rows"] = len(rows)

The file is written once on entry with status "running" and rewritten on exit
with status "complete" or "failed" plus the elapsed time. A sweep that crashes
halfway therefore leaves a partial CSV next to metadata that says so, which is
the case the sweeps' rewrite-after-every-run already anticipates.

TF32 is recorded as the runtime value of the two backend flags rather than as
the assumption that the script set them. Blackwell runs FP32 matmuls at 10
mantissa bits with TF32 on, so an FP32 baseline collected with it enabled is
not an FP32 baseline. That is worth an assertion in the record, not a comment.
"""

import contextlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def _git(*args, default=None):
    """Run a git command in the repo root; None if git or the repo is absent."""
    try:
        r = subprocess.run(("git",) + args, cwd=ROOT, capture_output=True,
                           text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return default
    # rstrip only. `git status --porcelain` encodes the staged/unstaged state
    # in the first two columns, so stripping the left would shift every path in
    # the first line by one character.
    return r.stdout.rstrip() if r.returncode == 0 else default


def _porcelain_path(line):
    """The path from one `git status --porcelain` line.

    Columns 0-1 are the status code and column 2 is a space. Renames read
    `R  old -> new`; the new name is the one that exists on disk.
    """
    path = line[2:].strip()
    return path.split(" -> ")[-1] if " -> " in path else path


def git_state():
    """Commit, branch, and what was uncommitted at the time of the run.

    `dirty` is the honest field. A record whose commit is clean can be
    checked out and rerun; one with dirty=true names the files that differed,
    so at least the discrepancy is visible instead of implied.
    """
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        # Paths only, capped: this is a record, not a diff.
        "dirty_files": sorted(map(_porcelain_path, status.splitlines()))[:50] if status else [],
    }


def machine():
    """Interpreter, torch build, GPU, and the numerical backend flags."""
    cuda = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
        "gpu_count": torch.cuda.device_count() if cuda else 0,
        # Read, not assumed. See the module docstring.
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def _jsonable(v):
    """Argparse namespaces hold Paths and tuples; JSON does not."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def describe_run(config=None, args=None, extra=None):
    """The metadata dict, without writing it.

    `config` is the module constants (the protocol), `args` the resolved
    argparse namespace (which captures defaults, unlike argv). Both are
    recorded because either alone is incomplete: argv omits every default,
    and the constants omit every override.
    """
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "running",
        "argv": [sys.executable] + sys.argv,
        "cwd": str(Path.cwd()),
        "git": git_state(),
        "config": _jsonable(config or {}),
        "args": _jsonable(vars(args) if args is not None else {}),
        "machine": machine(),
        **(_jsonable(extra) if extra else {}),
    }


def save_metadata(path, data):
    """Write a metadata file beside `path`, replacing its suffix with .meta.json."""
    out = Path(path).with_suffix(".meta.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return out


@contextlib.contextmanager
def record_run(csv_path, config=None, args=None, extra=None):
    """Write metadata for `csv_path`, then update it when the sweep ends.

    Yields the dict, so a caller can record anything only known at the end:

        with record_run(out, config=CONFIG, args=args) as meta:
            ...
            meta["rows"] = len(rows)

    Written on entry as well as exit so that a run killed partway still has
    metadata for the rows it did produce.
    """
    data = describe_run(config, args, extra)
    path = save_metadata(csv_path, data)
    print(f"run metadata -> {path}")
    t0 = time.time()
    try:
        yield data
    except BaseException as e:                  # KeyboardInterrupt included
        data["status"] = "failed"
        data["error"] = f"{type(e).__name__}: {e}"
        raise
    else:
        data["status"] = "complete"
    finally:
        data["elapsed_s"] = round(time.time() - t0, 1)
        data["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_metadata(csv_path, data)
