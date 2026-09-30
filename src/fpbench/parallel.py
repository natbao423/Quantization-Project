"""Spread a sweep's independent runs across GPUs, with a one-GPU fallback.

A sweep is a grid of runs that share nothing but the dataset, so it splits
cleanly: one worker process per GPU, each pulling the next run off a shared
list as soon as it finishes the last. Nothing is split statically, so a GPU
that is slower (the one driving the display, say) simply takes fewer runs.

    devices = resolve_devices(args.devices, args.jobs_per_gpu)
    with DevicePool(devices, setup, (act_at,)) as pool:
        for i, result in pool.imap(run_cell, cells):
            ...

`setup(device, *setup_args)` runs once per worker before any cell: it is where
the dataset gets loaded onto that worker's GPU and where the script points its
DEVICE global at it. `run_cell(cell)` does one run and returns something
picklable (rows, not a model). Both must be top-level functions of an
importable module, and the script itself counts: spawn re-imports it.

That re-import is also the trap. A worker gets the script's module constants as
written in the source, NOT as reassigned after argument parsing, because the
`if __name__ == "__main__"` block never runs there. Anything a flag changes has
to reach the worker through setup_args or the cell itself.

With a single device there are no workers at all. setup and every cell run in
this process, in order, exactly as the sweeps ran before this module existed.
That is the fallback on a one-GPU machine and the path `--devices 0` forces.

Results are independent of how the runs were spread. Every run seeds torch
itself, torch.manual_seed seeds every device, and the random streams used here
come out the same on any GPU of the same model. The sweeps write rows in grid
order, not completion order, so a parallel sweep's CSV lines up row for row
with a sequential one. (cuDNN nondeterminism is still there, as it always was.)
"""

import concurrent.futures as cf
import multiprocessing as mp

import torch

DEVICE_HELP = ("where to run: 'auto' (every discrete GPU, CPU if there are "
               "none), 'cpu', or GPU indices such as '0,1' or '1'")


def discrete_gpus():
    """([usable CUDA indices], [(index, name) skipped as integrated]).

    torch.cuda sees NVIDIA devices only, so an Intel or AMD integrated GPU
    never appears here to begin with; nothing needs filtering for the usual
    desktop iGPU. An NVIDIA integrated part (Jetson/Tegra-class, sharing system
    memory) does appear, reports is_integrated, and is skipped.

    Honours CUDA_VISIBLE_DEVICES, since device_count does.
    """
    if not torch.cuda.is_available():
        return [], []
    keep, skipped = [], []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        if p.is_integrated:
            skipped.append((i, p.name))
        else:
            keep.append(i)
    return keep, skipped


def _parse_ids(spec):
    ids = []
    for token in spec.replace(",", " ").split():
        i = int(token)
        if i < 0:
            raise ValueError(token)
        if i not in ids:
            ids.append(i)
    if not ids:
        raise ValueError(spec)
    return ids


def resolve_devices(spec="auto", jobs_per_gpu=1, log=print):
    """One device string per worker, e.g. ['cuda:0', 'cuda:1'] or ['cpu'].

    auto   every discrete GPU. One GPU found means one worker and runs go one at
           a time, as before; none found means the CPU, with a warning.
    cpu    the CPU, one worker.
    0,1    exactly these CUDA indices. A typo is an error, not a silent
           fallback, since an explicit request that quietly ran elsewhere
           would be recorded in the metadata as something it was not.

    jobs_per_gpu > 1 runs that many workers on each GPU. These models are small
    enough that one run does not fill a GPU, so two can overlap usefully; more
    than that mostly contends for the same SMs.
    """
    if jobs_per_gpu < 1:
        raise SystemExit(f"--jobs-per-gpu must be 1 or more, got {jobs_per_gpu}")
    spec = str(spec).strip().lower()
    if spec == "cpu":
        log("devices: cpu (as requested)")
        return ["cpu"]

    keep, skipped = discrete_gpus()
    if spec == "auto":
        for i, name in skipped:
            log(f"devices: skipping cuda:{i} ({name}), an integrated GPU")
        ids = keep
    else:
        try:
            ids = _parse_ids(spec)
        except ValueError:
            raise SystemExit(f"--devices: expected auto, cpu, or GPU indices "
                             f"like 0,1; got {spec!r}")
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n == 0:
            raise SystemExit("--devices: no CUDA device on this machine; "
                             "use --devices cpu")
        bad = [i for i in ids if i >= n]
        if bad:
            raise SystemExit(f"--devices: no CUDA device {', '.join(map(str, bad))}; "
                             f"this machine has {n} (indices 0-{n - 1})")
        for i, name in skipped:
            if i in ids:
                log(f"devices: cuda:{i} ({name}) is an integrated GPU; "
                    f"using it because it was asked for by index")

    if not ids:
        log("devices: no discrete CUDA GPU found, falling back to the CPU. "
            "Expect this to be many times slower.")
        return ["cpu"]

    names = {i: torch.cuda.get_device_name(i) for i in ids}
    listing = ", ".join(f"cuda:{i} ({names[i]})" for i in ids)
    jobs = f", {jobs_per_gpu} jobs each" if jobs_per_gpu > 1 else ""
    if len(ids) == 1:
        log(f"devices: 1 GPU, {listing}{jobs}"
            + ("" if jobs_per_gpu > 1 else "; runs go one at a time"))
    else:
        log(f"devices: {len(ids)} GPUs, {listing}{jobs}; runs are shared between them")
    return [f"cuda:{i}" for i in ids for _ in range(jobs_per_gpu)]


def gpu_count(devices):
    """Distinct GPUs in a resolve_devices list; 0 for the CPU."""
    return len({d for d in devices if d.startswith("cuda")})


def set_current(device):
    """Make `device` the current CUDA device, so a bare "cuda" means it too."""
    if device.startswith("cuda"):
        torch.cuda.set_device(device)


# Worker-side state. A failure in setup is kept rather than raised: raised in
# an initializer it breaks the pool with no message saying why, whereas kept
# here it is re-raised by the first cell, traceback and all.
_setup_error = None


def _init_worker(queue, setup, setup_args):
    global _setup_error
    try:
        device = queue.get(timeout=60)
        set_current(device)
        setup(device, *setup_args)
    except BaseException as e:            # noqa: BLE001 - reported by _call
        _setup_error = e


def _call(fn, item):
    if _setup_error is not None:
        raise RuntimeError("worker setup failed") from _setup_error
    return fn(item)


class DevicePool:
    """Run cells across the devices from resolve_devices. See module docstring."""

    def __init__(self, devices, setup, setup_args=()):
        self.devices = list(devices)
        self.setup, self.setup_args = setup, tuple(setup_args)
        self.executor = None

    @property
    def parallel(self):
        return len(self.devices) > 1

    def __enter__(self):
        if not self.parallel:
            set_current(self.devices[0])
            self.setup(self.devices[0], *self.setup_args)
            return self
        # spawn, not fork: CUDA cannot be used in a forked child, and spawn is
        # the only method on Windows anyway.
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        for d in self.devices:
            queue.put(d)
        self.executor = cf.ProcessPoolExecutor(
            max_workers=len(self.devices), mp_context=ctx,
            initializer=_init_worker, initargs=(queue, self.setup, self.setup_args))
        return self

    def imap(self, fn, items):
        """Yield (index, fn(item)) for every item, in completion order.

        Sequential on one device, so completion order is input order there. A
        worker that raises stops the sweep with its own traceback; one that
        dies outright (a driver crash) raises BrokenProcessPool rather than
        hanging, which is why this is not multiprocessing.Pool.
        """
        items = list(items)
        if not self.parallel:
            for i, item in enumerate(items):
                yield i, fn(item)
            return
        futures = {self.executor.submit(_call, fn, item): i
                   for i, item in enumerate(items)}
        for f in cf.as_completed(futures):
            yield futures[f], f.result()

    def map(self, fn, items):
        """fn over items, results in input order."""
        items = list(items)
        out = [None] * len(items)
        for i, r in self.imap(fn, items):
            out[i] = r
        return out

    def __exit__(self, exc_type, *_):
        if self.executor is not None:
            # On failure, drop the queued runs instead of finishing them all
            # first. A worker mid-run still completes that one run.
            self.executor.shutdown(wait=exc_type is None,
                                   cancel_futures=exc_type is not None)
            self.executor = None
        return False
