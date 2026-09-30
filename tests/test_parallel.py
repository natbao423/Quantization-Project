"""Device selection and the worker pool in fpbench.parallel.

Device detection is tested against a faked torch.cuda, so these run the same
on a two-GPU box, a one-GPU box and a machine with none. The pool itself is
tested for real, on CPU workers, since spawning is the part that can break.
"""

import os
from types import SimpleNamespace

import pytest

from fpbench import parallel
from fpbench.parallel import DevicePool, gpu_count, resolve_devices


def fake_gpus(monkeypatch, gpus):
    """Pretend torch.cuda sees `gpus`: a list of (name, is_integrated)."""
    cuda = parallel.torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: bool(gpus))
    monkeypatch.setattr(cuda, "device_count", lambda: len(gpus))
    monkeypatch.setattr(cuda, "get_device_name", lambda i: gpus[i][0])
    monkeypatch.setattr(cuda, "get_device_properties",
                        lambda i: SimpleNamespace(name=gpus[i][0],
                                                  is_integrated=gpus[i][1]))


TWO = [("RTX 5070 Ti", False), ("RTX 5070 Ti", False)]
ONE = [("RTX 5070 Ti", False)]


def resolve(*a, **kw):
    lines = []
    return resolve_devices(*a, log=lines.append, **kw), lines


# --------------------------------------------------------------------------
# resolve_devices
# --------------------------------------------------------------------------

def test_auto_uses_every_discrete_gpu(monkeypatch):
    fake_gpus(monkeypatch, TWO)
    devices, lines = resolve("auto")
    assert devices == ["cuda:0", "cuda:1"]
    assert "2 GPUs" in lines[-1]


def test_auto_falls_back_to_one_gpu(monkeypatch):
    fake_gpus(monkeypatch, ONE)
    devices, lines = resolve("auto")
    assert devices == ["cuda:0"]
    assert "one at a time" in lines[-1]


def test_auto_skips_an_integrated_gpu(monkeypatch):
    """An NVIDIA integrated part shows up in torch.cuda; it is not used."""
    fake_gpus(monkeypatch, [("Tegra iGPU", True), ("RTX 5070 Ti", False)])
    devices, lines = resolve("auto")
    assert devices == ["cuda:1"]
    assert any("skipping cuda:0" in l and "integrated" in l for l in lines)


def test_auto_with_only_an_integrated_gpu_falls_back_to_cpu(monkeypatch):
    fake_gpus(monkeypatch, [("Tegra iGPU", True)])
    devices, lines = resolve("auto")
    assert devices == ["cpu"]
    assert "falling back to the CPU" in lines[-1]


def test_auto_with_no_gpu_falls_back_to_cpu(monkeypatch):
    fake_gpus(monkeypatch, [])
    devices, lines = resolve("auto")
    assert devices == ["cpu"]
    assert "falling back to the CPU" in lines[-1]


def test_explicit_indices(monkeypatch):
    fake_gpus(monkeypatch, TWO)
    assert resolve("1")[0] == ["cuda:1"]
    assert resolve("1,0")[0] == ["cuda:1", "cuda:0"]
    assert resolve("0 1 0")[0] == ["cuda:0", "cuda:1"]     # duplicates dropped


def test_an_explicit_index_that_does_not_exist_is_an_error(monkeypatch):
    """Not a silent fallback: the metadata would record a device never used."""
    fake_gpus(monkeypatch, ONE)
    with pytest.raises(SystemExit, match="no CUDA device 1"):
        resolve("0,1")


def test_explicit_indices_with_no_gpu_is_an_error(monkeypatch):
    fake_gpus(monkeypatch, [])
    with pytest.raises(SystemExit, match="--devices cpu"):
        resolve("0")


@pytest.mark.parametrize("spec", ["gpu", "-1", "0,x", ""])
def test_a_malformed_spec_is_an_error(monkeypatch, spec):
    fake_gpus(monkeypatch, TWO)
    with pytest.raises(SystemExit, match="--devices"):
        resolve(spec)


def test_cpu_is_honoured_even_with_gpus(monkeypatch):
    fake_gpus(monkeypatch, TWO)
    assert resolve("CPU")[0] == ["cpu"]


def test_jobs_per_gpu_repeats_each_device(monkeypatch):
    fake_gpus(monkeypatch, TWO)
    devices, _ = resolve("auto", jobs_per_gpu=2)
    assert devices == ["cuda:0", "cuda:0", "cuda:1", "cuda:1"]
    assert gpu_count(devices) == 2


def test_jobs_per_gpu_must_be_positive(monkeypatch):
    fake_gpus(monkeypatch, TWO)
    with pytest.raises(SystemExit):
        resolve("auto", jobs_per_gpu=0)


def test_gpu_count():
    assert gpu_count(["cpu"]) == 0
    assert gpu_count(["cuda:0"]) == 1
    assert gpu_count(["cuda:0", "cuda:1"]) == 2


# --------------------------------------------------------------------------
# DevicePool. Worker functions must be module-level so spawn can import them.
# --------------------------------------------------------------------------

_STATE = {}


def setup(device, tag):
    _STATE.update(device=device, tag=tag, pid=os.getpid())


def work(item):
    return item * item, _STATE["device"], _STATE["tag"], _STATE["pid"]


def fail(item):
    if item == 2:
        raise ValueError("cell 2 is broken")
    return item


def broken_setup(device):
    raise RuntimeError("no dataset here")


def test_one_device_runs_in_this_process_in_order():
    """The one-GPU fallback: no workers, same process, input order."""
    _STATE.clear()
    with DevicePool(["cpu"], setup, ("t",)) as pool:
        assert not pool.parallel
        got = list(pool.imap(work, [3, 1, 2]))
    assert [i for i, _ in got] == [0, 1, 2]
    assert [r[0] for _, r in got] == [9, 1, 4]
    assert all(r[3] == os.getpid() for _, r in got)


def test_several_devices_share_the_work_across_processes():
    with DevicePool(["cpu", "cpu"], setup, ("t",)) as pool:
        assert pool.parallel
        got = dict(pool.imap(work, range(8)))
    assert sorted(got) == list(range(8))
    assert all(got[i][0] == i * i for i in range(8))
    assert all(got[i][2] == "t" for i in range(8))           # setup args arrived
    assert all(got[i][3] != os.getpid() for i in range(8))   # not in this process


def test_map_keeps_input_order():
    with DevicePool(["cpu", "cpu"], setup, ("t",)) as pool:
        assert [r[0] for r in pool.map(work, [5, 4, 3, 2, 1])] == [25, 16, 9, 4, 1]


def test_a_failing_cell_stops_the_sweep_with_its_own_error():
    with pytest.raises(ValueError, match="cell 2 is broken"):
        with DevicePool(["cpu", "cpu"], setup, ("t",)) as pool:
            list(pool.imap(fail, range(4)))


def test_a_failing_setup_is_reported_not_hung():
    with pytest.raises(RuntimeError, match="worker setup failed"):
        with DevicePool(["cpu", "cpu"], broken_setup) as pool:
            list(pool.imap(work, range(4)))
