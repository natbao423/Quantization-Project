"""Shared command-line surface for the sweep scripts.

Every sweep answers the same questions: which bit widths, which conditions,
which formats, how many seeds, where to write. Before this module each script
answered them differently or not at all. `train_at_vary_precision.py` had no
flags whatsoever, so its protocol could only be changed by editing constants --
precisely the thing `run_metadata` can report after the fact but not prevent.

Flags defined here mean the same thing in every script. Script-specific ones
(--act-at, --ptq, --steps, --batch-study) stay in the script that owns them.

    p = argparse.ArgumentParser()
    add_sweep_args(p, conditions=CONDITIONS, bits=[1, 2, 3, 23], seeds=3)
    args = p.parse_args()
    conditions = select_conditions(CONDITIONS, args.only)
    formats    = select_formats(args)
    out        = resolve_out(args, DATA_DIR, "mnist_cnn_curves.csv", args.only)
    guard_output(out, args.force)
"""

import argparse
import json
import pathlib

DEFAULT_BLOCK = 16
FORMAT_CHOICES = ("elementwise", "bfp")


def add_sweep_args(p, *, conditions, bits, seeds, formats=True,
                   block=DEFAULT_BLOCK):
    """Add the flags every sweep shares. `conditions` is for the help text."""
    names = [c[0] if isinstance(c, (tuple, list)) else c for c in conditions]

    p.add_argument("--bits", type=int, nargs="+", default=list(bits),
                   metavar="N",
                   help="mantissa widths to sweep (default: %(default)s). "
                        "23 means quantization off.")
    p.add_argument("--seeds", type=int, default=seeds, metavar="N",
                   help="number of seeds, run as range(N) (default: %(default)s)")
    p.add_argument("--only", nargs="+", default=None, metavar="COND",
                   help="limit the sweep to these conditions. One or more of: "
                        + ", ".join(names))
    if formats:
        p.add_argument("--formats", nargs="+", choices=FORMAT_CHOICES,
                       default=list(FORMAT_CHOICES), metavar="FMT",
                       help="number formats: elementwise (per-element "
                            "exponents, a scientific control) and/or bfp "
                            "(block floating point, the deployable one). "
                            "Default: both.")
        p.add_argument("--block", type=int, default=block, metavar="N",
                       help="BFP block size (default: %(default)s). Note the "
                            "CSV tag 'bfp16' means block 16, not bfloat16.")
    p.add_argument("--out", type=pathlib.Path, default=None, metavar="PATH",
                   help="output CSV path. Defaults to the canonical name for "
                        "this sweep, suffixed when --only narrows it.")
    p.add_argument("--force", action="store_true",
                   help="overwrite a results file that already looks complete")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and exit without training")
    return p


def select_conditions(available, only):
    """Filter `available` by --only, rejecting names that do not exist.

    An unrecognised name used to yield an empty condition list, which ran zero
    configurations, wrote zero rows and reported success. A typo should be an
    error, not a silent no-op.
    """
    names = [c[0] if isinstance(c, (tuple, list)) else c for c in available]
    if not only:
        return list(available)
    unknown = [o for o in only if o not in names]
    if unknown:
        raise SystemExit(
            f"unknown condition(s): {', '.join(unknown)}\n"
            f"available: {', '.join(names)}")
    keep = set(only)
    return [c for c, n in zip(available, names) if n in keep]


def select_formats(args):
    """[(tag, block)] from --formats/--block. block=None means per-element.

    The tag for BFP embeds the block size, so `bfp16` keeps meaning what it
    means in the committed CSVs while `--block 8` cannot silently pool with it.
    """
    if not hasattr(args, "formats"):
        return [("elementwise", None)]
    out = []
    for f in args.formats:
        if f == "elementwise":
            out.append(("elementwise", None))
        else:
            out.append((f"bfp{args.block}", args.block))
    return out


def resolve_out(args, directory, default_name, tag=None):
    """--out if given, else the canonical name, suffixed when --only narrows."""
    if args.out is not None:
        return pathlib.Path(args.out)
    stem, dot, ext = default_name.rpartition(".")
    suffix = "_" + "_".join(tag) if tag else ""
    return pathlib.Path(directory) / f"{stem}{suffix}{dot}{ext}"


def guard_output(path, force=False):
    """Refuse to clobber a results file that looks finished.

    The sweeps rewrite their CSV from scratch, so a short sanity run silently
    destroys a long one. That is not hypothetical: a 250-step smoke test
    overwrote a completed 4-hour transformer sweep, and only version control
    got it back.

    Blocks when the CSV exists AND either its metadata says the run completed
    or there is no metadata at all. The second case is the dangerous one, since
    every CSV committed before run_metadata existed has no sidecar. A file whose
    metadata says "failed" or "running" is a known-partial result and is allowed
    through, because rerunning it is the point.
    """
    path = pathlib.Path(path)
    if force or not path.exists():
        return

    meta = path.with_suffix(".meta.json")
    status, detail = None, "it has no .meta.json, so it predates run metadata"
    if meta.exists():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            status = data.get("status")
            detail = (f"its metadata says status={status!r}"
                      + (f", {data['rows']} rows" if "rows" in data else ""))
        except (OSError, ValueError):
            detail = "its .meta.json could not be read"

    if status in ("failed", "running"):
        print(f"note: overwriting {path.name}; {detail}")
        return

    raise SystemExit(
        f"refusing to overwrite {path}\n"
        f"  {detail}.\n"
        f"  Write elsewhere with --out PATH, or pass --force to overwrite.")


def print_plan(*, model, conditions, formats, bits, seeds, budget, out,
               seconds_per_run=None, extra=None):
    """--dry-run output: what would run, how much of it, and where it lands."""
    names = [c[0] if isinstance(c, (tuple, list)) else c for c in conditions]
    n = len(names) * len(formats) * len(bits) * seeds

    print(f"model       {model}")
    print(f"conditions  {', '.join(names)}")
    print(f"formats     {', '.join(f'{t} (block {b})' if b else t for t, b in formats)}")
    print(f"bits        {' '.join(str(b) for b in bits)}")
    print(f"seeds       {seeds}  (range(0, {seeds}))")
    print(f"budget      {budget}")
    for k, v in (extra or {}).items():
        print(f"{k:11s} {v}")
    print(f"runs        {n}")
    if seconds_per_run:
        secs = n * seconds_per_run
        print(f"estimate    ~{secs/60:.0f} min ({secs/3600:.1f} h) "
              f"at {seconds_per_run:.0f}s per run")
    print(f"output      {out}")
    exists = pathlib.Path(out).exists()
    print(f"            {'EXISTS, would need --force' if exists else 'new file'}")
