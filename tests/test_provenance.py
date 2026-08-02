"""Run-provenance tests.

The v1 pilot recorded no commit and no wall clock, so RUNS.md had to
reconstruct both from file mtimes afterwards, and the commit column in it is an
inference rather than a record. These tests pin the fields that make that
reconstruction unnecessary for every run from here on, including the `dirty`
flag: oneshot_prune_seed0 started three seconds before the launch script it
used was committed, so a commit hash on its own would have described that run
incorrectly while looking authoritative.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import (  # noqa: E402
    _prior_invocations,
    git_provenance,
    utc_now,
)

ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def test_utc_now_is_iso8601_utc_to_the_second():
    stamp = utc_now()
    assert ISO_Z.match(stamp), stamp
    # Parseable back, so a reader is not guessing at the format.
    datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")


def test_git_provenance_records_a_full_commit_hash_and_dirty_flag():
    prov = git_provenance()
    assert re.fullmatch(r"[0-9a-f]{40}", prov["commit"]), prov["commit"]
    assert isinstance(prov["dirty"], bool)
    assert prov["commit_time"] is not None


def test_git_provenance_outside_a_repo_returns_nulls_and_does_not_raise(tmp_path):
    """Provenance collection must never be able to abort a training run."""
    prov = git_provenance(tmp_path)
    assert prov["commit"] is None
    assert prov["dirty"] is None
    assert "note" in prov


def test_prior_invocations_is_empty_when_there_is_no_previous_run(tmp_path):
    assert _prior_invocations(tmp_path / "provenance.json") == []


def test_prior_invocations_carries_forward_the_earlier_start(tmp_path):
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps({
        "started_utc": "2026-07-19T20:34:58Z",
        "finished_utc": "2026-07-19T23:32:25Z",
        "git": {"commit": "a" * 40, "dirty": False},
    }))
    carried = _prior_invocations(path)
    assert len(carried) == 1
    assert carried[0]["started_utc"] == "2026-07-19T20:34:58Z"
    assert carried[0]["git"]["commit"] == "a" * 40


def test_prior_invocations_accumulate_across_several_resumes(tmp_path):
    """A run resumed twice keeps all three starts, not just the latest."""
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps({
        "started_utc": "2026-07-20T02:00:00Z",
        "finished_utc": None,
        "git": {"commit": "b" * 40, "dirty": False},
        "previous_invocations": [
            {"started_utc": "2026-07-19T20:00:00Z", "finished_utc": None,
             "git": {"commit": "a" * 40, "dirty": False}},
        ],
    }))
    carried = _prior_invocations(path)
    assert [c["started_utc"] for c in carried] == [
        "2026-07-19T20:00:00Z", "2026-07-20T02:00:00Z"]


def test_prior_invocations_records_a_commit_change_across_a_resume(tmp_path):
    """Resuming at a different commit is exactly what must not be overwritten."""
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps({
        "started_utc": "2026-07-20T02:00:00Z",
        "finished_utc": None,
        "git": {"commit": "a" * 40, "dirty": True},
    }))
    carried = _prior_invocations(path)
    assert carried[0]["git"]["commit"] == "a" * 40
    assert carried[0]["git"]["dirty"] is True
    assert carried[0]["finished_utc"] is None


def test_prior_invocations_tolerates_an_unreadable_file(tmp_path):
    path = tmp_path / "provenance.json"
    path.write_text("{ not json")
    assert _prior_invocations(path) == []


def test_prior_invocations_ignores_a_file_without_a_start(tmp_path):
    """v1 provenance files have no started_utc and carry no invocation record."""
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps({"run_id": "rigl_seed0", "device": "cuda:0"}))
    assert _prior_invocations(path) == []
