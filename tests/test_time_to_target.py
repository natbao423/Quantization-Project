"""Time-to-target at sub-epoch resolution.

Before --evals-per-epoch, time-to-98% was counted in whole epochs, and every
gradient configuration - FP32 included - read 3. An epoch is 430 optimizer
steps, so the measurement could not see any difference smaller than that.
"""

import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_curves as S      # noqa: E402
import train_mnist_cnn as T       # noqa: E402


# --------------------------------------------------------------------------
# where mid-epoch evaluations happen
# --------------------------------------------------------------------------

def test_one_eval_per_epoch_is_the_old_behaviour():
    assert T.mid_epoch_evals(430, 1) == set()


def test_ten_per_epoch_are_evenly_spaced_and_skip_the_last_batch():
    """The end-of-epoch evaluation already covers the last batch."""
    pts = sorted(T.mid_epoch_evals(430, 10))
    assert len(pts) == 9
    assert pts[-1] < 429
    assert {b - a for a, b in zip(pts, pts[1:])} == {43}


def test_more_evals_than_batches_means_every_batch_but_the_last():
    assert T.mid_epoch_evals(5, 50) == {0, 1, 2, 3}


# --------------------------------------------------------------------------
# first-hit bookkeeping
# --------------------------------------------------------------------------

def test_records_the_first_hit_and_ignores_later_dips():
    """First-hitting time: dropping back below a target does not undo it, and
    a later, higher reading does not move it."""
    hits = {0.90: None, 0.98: None}
    T.record_crossings(hits, 0.4, 0.93)
    T.record_crossings(hits, 0.5, 0.85)
    T.record_crossings(hits, 2.3, 0.981)
    T.record_crossings(hits, 2.4, 0.990)
    assert hits == {0.90: 0.4, 0.98: 2.3}


def test_a_target_never_reached_stays_none():
    hits = {0.98: None}
    for epoch in range(1, 13):
        T.record_crossings(hits, float(epoch), 0.97)
    assert hits[0.98] is None


def test_the_two_scripts_agree_on_the_targets():
    """The sweep writes epochs_to_* columns and the summarizer reads them;
    if the target lists drifted apart a table would silently go blank."""
    assert tuple(S.TARGET_ACCS) == T.TIME_TO
    assert S.FINE == [f"epochs_to_{round(t * 100)}" for t in T.TIME_TO]


# --------------------------------------------------------------------------
# the summarizer
# --------------------------------------------------------------------------

def _curve(fine=None):
    rows = [{"epoch": e, "val_acc": a, "val_loss": 0.1}
            for e, a in ((1, 0.95), (2, 0.975), (3, 0.985))]
    if fine:
        for r in rows:
            r.update(fine)
    return rows


def test_summarizer_uses_the_sub_epoch_value_when_present():
    fine = {"epochs_to_90": 0.3, "epochs_to_95": 1.2, "epochs_to_98": 2.4}
    out = S.per_run(_curve(fine), [], fine=True)
    assert (out["ep_to_90"], out["ep_to_95"], out["ep_to_98"]) == (0.3, 1.2, 2.4)


def test_summarizer_falls_back_to_whole_epochs_for_old_files():
    out = S.per_run(_curve(), [], fine=False)
    assert (out["ep_to_90"], out["ep_to_95"], out["ep_to_98"]) == (1, 1, 3)


def test_a_blank_first_hit_means_never_not_zero():
    fine = {"epochs_to_90": 0.3, "epochs_to_95": 1.2, "epochs_to_98": None}
    assert S.per_run(_curve(fine), [], fine=True)["ep_to_98"] is None


def test_load_detects_first_hit_columns(tmp_path):
    base = ["format", "target", "bits", "seed", "epoch", "val_acc", "val_loss"]
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    with old.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(base)
        w.writerow(["elementwise", "grad", 4, 0, 1, 0.99, 0.1])
    with new.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(base + S.FINE)
        w.writerow(["elementwise", "grad", 4, 0, 1, 0.99, 0.1, 0.3, 0.8, ""])
    assert S.load(old)[2] is False
    runs, _, fine = S.load(new)
    assert fine is True
    row = runs[("elementwise", "grad", 4, 0)][0]
    assert row["epochs_to_95"] == 0.8 and row["epochs_to_98"] is None
