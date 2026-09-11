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
import sys
import time

DEFAULT_BLOCK = 16
FORMAT_CHOICES = ("elementwise", "bfp")


BITS_MIN, BITS_MAX = 1, 23


def mantissa_bits(text):
    """argparse type: a whole number of mantissa bits, 1 to 23.

    Without this, --bits accepted anything: 30 silently quantized nothing (the
    quantizers treat 23 and above as FP32) and -1 crashed deep inside torch.
    """
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number")
    if not BITS_MIN <= v <= BITS_MAX:
        raise argparse.ArgumentTypeError(
            f"{v} is outside {BITS_MIN}-{BITS_MAX} "
            f"({BITS_MAX} is FP32; there are no more mantissa bits than that)")
    return v


def parse_bits(line):
    """'4', '2 4 7' or '2,4,7' -> a list of widths, duplicates removed.

    Duplicates matter: a width given twice runs twice under the same seeds, and
    the summarizer then merges the two runs' epochs into one corrupted curve.
    """
    widths = []
    for token in line.replace(",", " ").split():
        v = mantissa_bits(token)
        if v not in widths:
            widths.append(v)
    if not widths:
        raise argparse.ArgumentTypeError("no widths given")
    return widths


def seed_count(text):
    """argparse type: a whole number of seeds, at least 1."""
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number")
    if v < 1:
        raise argparse.ArgumentTypeError(f"{v} seeds would run nothing; use 1 or more")
    return v


def _ask(question, parse, default, default_text, what, read):
    """Ask until a valid answer; Enter keeps the default, Ctrl+C cancels."""
    prompt = f"{question}\nPress Enter for the default [{default_text}]: "
    while True:
        try:
            line = read(prompt)
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(f"\ncancelled: no {what} chosen")
        if not line.strip():
            return default
        try:
            return parse(line)
        except argparse.ArgumentTypeError as e:
            print(f"  {e}. Try again.", flush=True)


def ask_bits(default, read=input):
    """Ask for mantissa widths in the terminal."""
    return _ask(f"Mantissa bits to run, {BITS_MIN}-{BITS_MAX}: one number (4) or "
                f"several (2 4 7). {BITS_MAX} is the FP32 baseline.",
                parse_bits, list(default), " ".join(map(str, default)),
                "mantissa bits", read)


def ask_seeds(default, runs_per_seed=None, read=input):
    """Ask for a seed count, saying what each seed costs when that is known.

    Asked after the widths, so the cost per seed reflects the widths just
    chosen - it is the number that decides how long the run takes.
    """
    cost = f" Each seed is {runs_per_seed} runs here." if runs_per_seed else ""
    return _ask(f"Number of seeds, 1 or more: every configuration is repeated "
                f"once per seed, from a different random start.{cost}",
                lambda line: seed_count(line.strip()), default, str(default),
                "seed count", read)


def stdin_is_terminal(stream=None):
    """True only when someone can actually type an answer.

    isatty() alone is not enough on Windows: the NUL device is a character
    device and reports True, so a run fed from NUL would print the question,
    read end-of-file and exit. A real console is the only stdin that
    GetConsoleMode accepts, so that is the test there. (Git Bash's mintty
    window is a pipe, not a console, so it gets the default rather than the
    question; pass --bits there.)
    """
    stream = sys.stdin if stream is None else stream
    try:
        if stream is None or not stream.isatty():
            return False
        if sys.platform != "win32":
            return True
        import ctypes
        import msvcrt
        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = ctypes.c_uint32()
        return bool(ctypes.windll.kernel32.GetConsoleMode(
            ctypes.c_void_p(handle), ctypes.byref(mode)))
    except (AttributeError, OSError, ValueError):
        return False


def resolve_bits(parser, args, prompt=True, interactive=None, read=input):
    """Fill in args.bits: as given, else asked for, else the sweep's default.

    Asking happens only when prompt is true (the mode actually trains at each
    width) and stdin is a real terminal, so a background run or a pipe never
    sits waiting for an answer nobody can give. Whatever the source, the
    resolved list is what the run metadata records under args.
    """
    default = parser.sweep_bits_default
    interactive = stdin_is_terminal() if interactive is None else interactive
    if getattr(args, "bits", None) is not None:
        args.bits = parse_bits(" ".join(map(str, args.bits)))
    elif prompt and interactive:
        args.bits = ask_bits(default, read)
    else:
        args.bits = list(default)
    return args.bits



def resolve_seeds(parser, args, prompt=True, interactive=None, read=input,
                  runs_per_seed=None):
    """Fill in args.seeds: as given, else asked for, else the sweep's default.

    Same rules as resolve_bits: ask only in modes that train once per seed and
    only when someone can type an answer.
    """
    interactive = stdin_is_terminal() if interactive is None else interactive
    if getattr(args, "seeds", None) is None:
        args.seeds = (ask_seeds(parser.sweep_seeds_default, runs_per_seed, read)
                      if prompt and interactive else parser.sweep_seeds_default)
    return args.seeds

def add_sweep_args(p, *, conditions, bits, seeds, formats=True,
                   block=DEFAULT_BLOCK):
    """Add the flags every sweep shares. `conditions` is for the help text."""
    names = [c[0] if isinstance(c, (tuple, list)) else c for c in conditions]

    # SUPPRESS rather than a default list, so resolve_bits() can tell "not
    # given" (ask, or fall back to the default) from "given".
    p.add_argument("--bits", type=mantissa_bits, nargs="+", default=argparse.SUPPRESS,
                   metavar="N",
                   help=f"mantissa widths to run, each {BITS_MIN}-{BITS_MAX}; "
                        f"{BITS_MAX} means FP32, no quantization. Leave it out to "
                        f"be asked in the terminal. When there is no terminal to "
                        f"ask in (a background run, a pipe) the default is used: "
                        f"{' '.join(map(str, bits))}")
    p.sweep_bits_default = list(bits)
    p.add_argument("--seeds", type=seed_count, default=argparse.SUPPRESS, metavar="N",
                   help=f"number of seeds, 1 or more, run as range(N). Leave it "
                        f"out to be asked in the terminal; with no terminal the "
                        f"default is used: {seeds}")
    p.sweep_seeds_default = seeds
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
               seconds_per_run=None, extra=None, runs=None):
    """--dry-run output: what would run, how much of it, and where it lands.

    `runs` overrides the conditions x formats x bits x seeds count, for a mode
    that varies something else - the batch study multiplies by batch sizes.
    """
    names = [c[0] if isinstance(c, (tuple, list)) else c for c in conditions]
    n = runs if runs is not None else len(names) * len(formats) * len(bits) * seeds

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


def human_time(seconds):
    """Compact duration: 45s, 12m, 1.4h."""
    if seconds is None:
        return "?"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


class Progress:
    """Run counter, bar and ETA for a sweep.

    Every print here is flushed explicitly. Python only line-buffers stdout when
    it is attached to a terminal; redirected to a file or a pipe it switches to
    a block buffer, so a sweep that prints one short line per run shows NOTHING
    for the first several thousand characters. On these sweeps that is many
    minutes of apparent silence with a healthy job underneath, which is
    indistinguishable from a hang.

    The ETA is measured, not the static per-run estimate --dry-run prints. It
    divides real elapsed time by completed runs, so it self-corrects for a
    slower machine, a warm-up run, or a condition that costs more than the rest.
    Early estimates are unreliable for exactly that reason and are shown as `?`
    until the first run lands.
    """

    def __init__(self, total, width=10, stream=None):
        self.total = max(int(total), 1)
        self.done = 0
        self.width = width
        self.stream = stream or sys.stdout
        self.t0 = time.time()

    @property
    def elapsed(self):
        return time.time() - self.t0

    @property
    def eta(self):
        """Seconds remaining, or None before anything has finished."""
        if self.done == 0:
            return None
        return self.elapsed / self.done * (self.total - self.done)

    def bar(self):
        filled = round(self.width * self.done / self.total)
        return "#" * filled + "-" * (self.width - filled)

    def prefix(self):
        pct = 100 * self.done / self.total
        w = len(str(self.total))
        return f"[{self.done:{w}d}/{self.total} {pct:3.0f}%|{self.bar()}]"

    def state(self):
        """Machine-readable progress, for the run metadata sidecar.

        Written to disk after every run, so a sweep's progress is readable from
        another terminal even when its stdout is block-buffered into a log or
        the terminal that launched it is gone.
        """
        return {"done": self.done, "total": self.total,
                "pct": round(100 * self.done / self.total, 1),
                "elapsed_s": round(self.elapsed, 1),
                "eta_s": None if self.eta is None else round(self.eta, 1)}

    def note(self, message):
        """A flushed line that does not advance the counter."""
        print(message, file=self.stream, flush=True)

    def step(self, message):
        """Count one completed run and print it with progress and ETA."""
        self.done += 1
        print(f"{self.prefix()} {message}  eta {human_time(self.eta)}",
              file=self.stream, flush=True)

    def done_line(self, what="runs"):
        self.note(f"{self.done} {what} in {human_time(self.elapsed)}")


class Phase:
    """Announce a slow step that would otherwise be silent, and time it.

        with Phase("loading MNIST onto the GPU"):
            ...

    Loading MNIST stacks 55,000 images one at a time and takes long enough to
    look like a hang, with no output of its own to give it away.
    """

    def __init__(self, label, stream=None):
        self.label = label
        self.stream = stream or sys.stdout

    def __enter__(self):
        self.t0 = time.time()
        print(f"{self.label} ...", end="", file=self.stream, flush=True)
        return self

    def __exit__(self, *exc):
        status = "failed" if exc[0] else f"done in {human_time(time.time() - self.t0)}"
        print(f" {status}", file=self.stream, flush=True)
        return False
