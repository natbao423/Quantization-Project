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


# --------------------------------------------------------------------------
# progress reporting
# --------------------------------------------------------------------------

from fpbench.cli import Phase, Progress, human_time


@pytest.mark.parametrize("secs,want", [
    (0, "0s"), (45, "45s"), (89, "89s"), (90, "2m"), (600, "10m"),
    (5399, "90m"), (5400, "1.5h"), (7200, "2.0h"), (None, "?"),
])
def test_human_time(secs, want):
    assert human_time(secs) == want


def test_progress_has_no_eta_before_the_first_run():
    """Dividing elapsed by zero completed runs is not an estimate."""
    assert Progress(10).eta is None
    assert "?" in human_time(Progress(10).eta)


def test_progress_eta_extrapolates_from_measured_time():
    p = Progress(100)
    p.t0 -= 60.0                    # pretend a minute has passed
    p.done = 20
    assert 235 < p.eta < 245        # 3s/run x 80 remaining


def test_progress_bar_fills_and_reaches_100_percent():
    p = Progress(4, width=8)
    assert p.bar() == "-" * 8
    for _ in range(4):
        p.done += 1
    assert p.bar() == "#" * 8
    assert "100%" in p.prefix()


def test_progress_counts_every_step(capsys):
    p = Progress(3)
    for i in range(3):
        p.step(f"run {i}")
    out = capsys.readouterr().out
    assert out.count("\n") == 3
    assert "[3/3 100%" in out
    assert p.done == 3


def test_progress_state_is_json_shaped():
    """The sidecar is the progress source when stdout is buffered away."""
    p = Progress(50)
    p.t0 -= 10.0
    p.done = 10
    st = p.state()
    assert st["done"] == 10 and st["total"] == 50 and st["pct"] == 20.0
    assert st["eta_s"] > 0
    json.dumps(st)                  # must survive the metadata writer


def test_phase_reports_completion_and_reraises(capsys):
    with Phase("loading"):
        pass
    assert "loading ... done in" in capsys.readouterr().out

    with pytest.raises(ValueError):
        with Phase("loading"):
            raise ValueError("boom")
    assert "failed" in capsys.readouterr().out


# --------------------------------------------------------------------------
# choosing mantissa bits
# --------------------------------------------------------------------------

from fpbench.cli import ask_bits, mantissa_bits, parse_bits, resolve_bits


@pytest.mark.parametrize("text,want", [("1", 1), ("4", 4), ("23", 23), (" 7 ", 7)])
def test_mantissa_bits_accepts_1_to_23(text, want):
    assert mantissa_bits(text) == want


@pytest.mark.parametrize("text", ["0", "24", "30", "-1", "four", "4.5", ""])
def test_mantissa_bits_rejects_everything_else(text):
    """30 used to be accepted and silently quantize nothing; -1 crashed in torch."""
    with pytest.raises(argparse.ArgumentTypeError):
        mantissa_bits(text)


@pytest.mark.parametrize("line,want", [
    ("4", [4]), ("2 4 7", [2, 4, 7]), ("2,4, 7", [2, 4, 7]), ("7 2", [7, 2]),
])
def test_parse_bits_reads_one_or_several(line, want):
    assert parse_bits(line) == want


def test_parse_bits_drops_duplicates():
    """A repeated width would run twice under the same seeds, and the
    summarizer would merge the two runs into one corrupted curve."""
    assert parse_bits("4 4 2 4") == [4, 2]


def test_bits_out_of_range_is_rejected_on_the_command_line():
    p = argparse.ArgumentParser()
    add_sweep_args(p, conditions=CONDITIONS, bits=[1, 23], seeds=3)
    with pytest.raises(SystemExit):
        p.parse_args(["--bits", "30"])


def _parser():
    p = argparse.ArgumentParser()
    add_sweep_args(p, conditions=CONDITIONS, bits=[1, 2, 23], seeds=3)
    return p


def _answers(*lines):
    """A stand-in for input() that replays typed answers."""
    it = iter(lines)
    return lambda prompt="": next(it)


def test_given_bits_are_used_and_never_asked():
    p = _parser()
    args = p.parse_args(["--bits", "4", "4", "2"])
    never = lambda prompt="": pytest.fail("asked despite --bits")
    assert resolve_bits(p, args, interactive=True, read=never) == [4, 2]


def test_no_terminal_means_the_default_and_no_question():
    """A background run or a pipe must never sit waiting for an answer."""
    p = _parser()
    args = p.parse_args([])
    never = lambda prompt="": pytest.fail("asked with no terminal")
    assert resolve_bits(p, args, interactive=False, read=never) == [1, 2, 23]


def test_modes_that_do_not_sweep_never_ask():
    p = _parser()
    args = p.parse_args([])
    never = lambda prompt="": pytest.fail("asked in a non-sweep mode")
    assert resolve_bits(p, args, prompt=False, interactive=True, read=never) == [1, 2, 23]


def test_typing_a_number_selects_it():
    p = _parser()
    args = p.parse_args([])
    assert resolve_bits(p, args, interactive=True, read=_answers("4")) == [4]
    assert args.bits == [4]


def test_enter_keeps_the_default():
    assert ask_bits([1, 2, 23], read=_answers("")) == [1, 2, 23]


def test_a_bad_answer_is_asked_again(capsys):
    assert ask_bits([1, 23], read=_answers("30", "abc", "5")) == [5]
    out = capsys.readouterr().out
    assert out.count("Try again") == 2


def test_closing_the_prompt_exits_cleanly():
    def eof(prompt=""):
        raise EOFError
    with pytest.raises(SystemExit, match="cancelled"):
        ask_bits([1, 23], read=eof)


def test_the_null_device_is_not_a_terminal():
    """On Windows NUL reports isatty() True; it must still not count, or a
    run fed from it prints the question, reads end-of-file and exits."""
    import os
    from fpbench.cli import stdin_is_terminal
    with open(os.devnull) as nul:
        assert stdin_is_terminal(nul) is False


# --------------------------------------------------------------------------
# choosing the number of seeds
# --------------------------------------------------------------------------

from fpbench.cli import ask_seeds, resolve_seeds, seed_count


@pytest.mark.parametrize("text,want", [("1", 1), ("3", 3), ("10", 10), (" 5 ", 5)])
def test_seed_count_accepts_positive_whole_numbers(text, want):
    assert seed_count(text) == want


@pytest.mark.parametrize("text", ["0", "-2", "three", "2.5", "3 4", ""])
def test_seed_count_rejects_everything_else(text):
    """Zero seeds would run nothing and report success."""
    with pytest.raises(argparse.ArgumentTypeError):
        seed_count(text)


def test_seeds_of_zero_is_rejected_on_the_command_line():
    with pytest.raises(SystemExit):
        _parser().parse_args(["--seeds", "0"])


def test_given_seeds_are_used_and_never_asked():
    p = _parser()
    args = p.parse_args(["--seeds", "5"])
    never = lambda prompt="": pytest.fail("asked despite --seeds")
    assert resolve_seeds(p, args, interactive=True, read=never) == 5


def test_seeds_default_without_a_terminal():
    p = _parser()
    args = p.parse_args([])
    never = lambda prompt="": pytest.fail("asked with no terminal")
    assert resolve_seeds(p, args, interactive=False, read=never) == 3


def test_seeds_are_not_asked_in_modes_that_do_not_sweep():
    p = _parser()
    args = p.parse_args([])
    never = lambda prompt="": pytest.fail("asked in a non-sweep mode")
    assert resolve_seeds(p, args, prompt=False, interactive=True, read=never) == 3


def test_typing_a_seed_count_selects_it():
    p = _parser()
    args = p.parse_args([])
    assert resolve_seeds(p, args, interactive=True, read=_answers("7")) == 7
    assert args.seeds == 7


def test_enter_keeps_the_default_seed_count():
    assert ask_seeds(3, read=_answers("")) == 3


def test_a_bad_seed_count_is_asked_again(capsys):
    assert ask_seeds(3, read=_answers("0", "lots", "4")) == 4
    assert capsys.readouterr().out.count("Try again") == 2


def test_the_seed_question_quotes_the_cost_per_seed():
    """Asked after the widths, so the reader sees what a seed costs."""
    seen = []
    ask_seeds(3, runs_per_seed=48, read=lambda prompt="": seen.append(prompt) or "")
    assert "48 runs" in seen[0]


def test_bits_then_seeds_in_one_session():
    """The order a user meets them: widths first, then how many seeds."""
    p = _parser()
    args = p.parse_args([])
    typed = _answers("4 23", "5")
    resolve_bits(p, args, interactive=True, read=typed)
    resolve_seeds(p, args, interactive=True, read=typed)
    assert (args.bits, args.seeds) == ([4, 23], 5)
