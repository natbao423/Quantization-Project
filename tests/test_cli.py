"""Tests for the shared sweep command-line surface.

Two of these guard against mistakes that have actually happened in this repo: a
250-step sanity run overwrote a completed four-hour transformer sweep, and a
mistyped --only silently swept zero configurations and reported success.
"""

import argparse
import json

import pytest

from fpbench.cli import (add_sweep_args, guard_output, resolve_out,
                         select_conditions, select_formats)

CONDITIONS = [("input", True, False), ("weight", False, True),
              ("both", True, True)]


def parse(*argv, conditions=CONDITIONS, formats=True):
    p = argparse.ArgumentParser()
    add_sweep_args(p, conditions=conditions, bits=[1, 23], seeds=3,
                   formats=formats)
    return p.parse_args(list(argv))


# --------------------------------------------------------------------------
# conditions
# --------------------------------------------------------------------------

def test_no_only_keeps_every_condition():
    assert select_conditions(CONDITIONS, None) == CONDITIONS


def test_only_filters_and_preserves_declared_order():
    got = select_conditions(CONDITIONS, ["both", "input"])
    assert [c[0] for c in got] == ["input", "both"]


def test_unknown_condition_is_an_error_not_an_empty_sweep():
    """A typo used to yield zero conditions, zero runs, and a success message."""
    with pytest.raises(SystemExit) as e:
        select_conditions(CONDITIONS, ["wieght"])
    assert "wieght" in str(e.value) and "weight" in str(e.value)


def test_plain_string_conditions_work_too():
    assert select_conditions(["a", "b"], ["b"]) == ["b"]


# --------------------------------------------------------------------------
# formats
# --------------------------------------------------------------------------

def test_default_formats_are_both():
    assert select_formats(parse()) == [("elementwise", None), ("bfp16", 16)]


def test_block_size_is_in_the_tag_so_runs_cannot_pool():
    """bfp8 and bfp16 must not share a tag; the CSV column is the only thing
    keeping two block sizes apart in a pooled results file."""
    assert select_formats(parse("--formats", "bfp", "--block", "8")) == [("bfp8", 8)]


def test_formats_absent_falls_back_to_elementwise():
    assert select_formats(parse(formats=False)) == [("elementwise", None)]


# --------------------------------------------------------------------------
# output paths
# --------------------------------------------------------------------------

def test_default_output_is_the_canonical_name(tmp_path):
    assert resolve_out(parse(), tmp_path, "curves.csv").name == "curves.csv"


def test_only_suffixes_the_output_name(tmp_path):
    out = resolve_out(parse(), tmp_path, "curves.csv", ["weight", "input"])
    assert out.name == "curves_weight_input.csv"


def test_explicit_out_wins(tmp_path):
    args = parse("--out", str(tmp_path / "x.csv"))
    assert resolve_out(args, tmp_path, "curves.csv", ["weight"]).name == "x.csv"


# --------------------------------------------------------------------------
# the overwrite guard
# --------------------------------------------------------------------------

def test_missing_file_is_fine(tmp_path):
    guard_output(tmp_path / "nope.csv")


def test_refuses_a_file_with_no_metadata(tmp_path):
    """The dangerous case: every CSV committed before run_metadata existed has
    no sidecar, and those are the results worth protecting."""
    csv = tmp_path / "curves.csv"
    csv.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        guard_output(csv)


def test_refuses_a_completed_run(tmp_path):
    csv = tmp_path / "curves.csv"
    csv.write_text("a,b\n1,2\n")
    csv.with_suffix(".meta.json").write_text(
        json.dumps({"status": "complete", "rows": 2304}))
    with pytest.raises(SystemExit, match="complete"):
        guard_output(csv)


@pytest.mark.parametrize("status", ["failed", "running"])
def test_allows_a_partial_run(tmp_path, status):
    """Rerunning a crashed sweep is the normal case and must not need --force."""
    csv = tmp_path / "curves.csv"
    csv.write_text("a,b\n1,2\n")
    csv.with_suffix(".meta.json").write_text(json.dumps({"status": status}))
    guard_output(csv)


def test_force_overrides(tmp_path):
    csv = tmp_path / "curves.csv"
    csv.write_text("a,b\n1,2\n")
    guard_output(csv, force=True)


def test_unreadable_metadata_still_refuses(tmp_path):
    """A corrupt sidecar is not evidence that overwriting is safe."""
    csv = tmp_path / "curves.csv"
    csv.write_text("a,b\n1,2\n")
    csv.with_suffix(".meta.json").write_text("{not json")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        guard_output(csv)
