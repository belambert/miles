"""Sandbox lifecycle events: store round trip, aggregation, endpoint and report hook."""

import asyncio
import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from miles.dashboard import backend, hooks
from miles.dashboard.collector import CollectorConfig, DashboardCollector
from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.sandbox_status import MAX_FAILURE_BUCKETS, sandbox_status
from miles.dashboard.server import make_app
from miles.dashboard.store import MetricStore, SandboxEvent, SandboxStage, Stream
from miles.rollout.agentic.agent_function import call_agent_function
from miles.utils.test_utils import sandbox_event_stubs


def _event(ts: float, trial_id: str, stage: SandboxStage, exception_type: str | None = None) -> SandboxEvent:
    return SandboxEvent(ts=ts, trial_id=trial_id, task="t", stage=stage.value, exception_type=exception_type)


EVENTS = [
    _event(0.0, "a", SandboxStage.CREATED),
    _event(5.0, "a", SandboxStage.AGENT_STARTED),
    _event(1.0, "b", SandboxStage.CREATED),
    _event(70.0, "b", SandboxStage.FAILED, "TimeoutError"),
    _event(2.0, "c", SandboxStage.CREATED),
    _event(130.0, "c", SandboxStage.FAILED, "TimeoutError"),
    _event(3.0, "d", SandboxStage.CREATED),
    _event(75.0, "d", SandboxStage.FAILED),
    _event(4.0, "e", SandboxStage.TORN_DOWN),
    _event(200.0, "f", SandboxStage.SETUP_DONE),
]


def test_stage_counts_use_each_trials_latest_event_regardless_of_arrival_order():
    status = sandbox_status(list(reversed(EVENTS)), now=200.0, stale_after_s=600.0, bucket_s=60.0)
    assert status["num_trials"] == 6
    counts = status["stage_counts"]
    assert (counts["agent_started"], counts["failed"], counts["torn_down"], counts["setup_done"]) == (1, 3, 1, 1)
    assert counts["created"] == 0, "a superseded stage must not be counted"


def test_failures_are_bucketed_per_exception_type_with_a_missing_type_as_unknown():
    status = sandbox_status(EVENTS, now=200.0, stale_after_s=600.0, bucket_s=60.0)
    assert status["failure_counts"] == {"TimeoutError": 2, "unknown": 1}
    series = {s["label"]: s for s in status["failure_series"]}
    assert series["TimeoutError"]["ts"] == [70.0, 130.0]
    assert series["TimeoutError"]["value"] == [1, 1]
    assert series["unknown"]["value"] == [1, 0]


def test_only_silent_non_terminal_trials_are_stale():
    status = sandbox_status(EVENTS, now=200.0, stale_after_s=60.0, bucket_s=60.0)
    assert [row["trial_id"] for row in status["stale"]] == ["a"]
    assert status["stale"][0]["silent_s"] == 195.0


def test_a_stale_row_keeps_the_labels_of_the_trials_earlier_events():
    created = SandboxEvent(ts=0.0, trial_id="a", task="t", stage="created", provider="modal", sandbox_id="sb-1")
    started = SandboxEvent(ts=5.0, trial_id="a", task="", stage="agent_started")
    [row] = sandbox_status([created, started], now=100.0, stale_after_s=60.0, bucket_s=60.0)["stale"]
    assert (row["stage"], row["task"], row["provider"], row["sandbox_id"]) == ("agent_started", "t", "modal", "sb-1")


def test_a_tiny_bucket_is_widened_to_bound_the_bucket_count():
    status = sandbox_status(EVENTS, now=200.0, stale_after_s=600.0, bucket_s=1e-6)
    for series in status["failure_series"]:
        assert len(series["value"]) <= MAX_FAILURE_BUCKETS + 1
    assert sum(sum(series["value"]) for series in status["failure_series"]) == 3


@pytest.fixture
def dump_dir(tmp_path: Path) -> Path:
    store = MetricStore(tmp_path / "dashboard")
    for event in EVENTS:
        store.append(event)
    store.flush()
    return tmp_path


def test_static_endpoint_judges_staleness_at_the_last_recorded_event(dump_dir: Path):
    client = TestClient(make_app(MetricStore.load(dump_dir / "dashboard"), DumpReader(dump_dir)))
    assert client.get("/api/meta").json()["capabilities"]["has_sandbox_events"] is True
    status = client.get("/api/sandboxes", params=dict(stale_after_s=60)).json()
    assert status["now"] == 200.0
    assert [row["trial_id"] for row in status["stale"]] == ["a"]
    for bad in (0, "inf", "nan"):
        assert client.get("/api/sandboxes", params=dict(bucket_s=bad)).status_code == 400
    assert client.get("/api/sandboxes", params=dict(stale_after_s="inf")).status_code == 400


def test_follow_endpoint_judges_staleness_at_wall_clock(dump_dir: Path):
    client = TestClient(make_app(MetricStore.load(dump_dir / "dashboard"), DumpReader(dump_dir), follow=True))
    stale = client.get("/api/sandboxes").json()["stale"]
    assert {row["trial_id"] for row in stale} == {"a", "f"}


class _FakeRemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, *args):
        self.calls.append(args)


class _FakeHandle:
    def __init__(self):
        self.push_sandbox_events = _FakeRemoteMethod()


def test_report_is_a_noop_without_a_collector(monkeypatch):
    monkeypatch.setattr(backend, "_handle", None)
    hooks.report_sandbox_event("a", SandboxStage.CREATED)


def test_report_pushes_to_the_collector_and_swallows_a_bad_stage(monkeypatch):
    handle = _FakeHandle()
    monkeypatch.setattr(backend, "_handle", handle)
    hooks.report_sandbox_event("a", SandboxStage.FAILED, task="t", exception_type="OSError", provider="modal")
    hooks.report_sandbox_event("a", "not-a-stage")
    [([event],)] = handle.push_sandbox_events.calls
    assert (event.trial_id, event.stage, event.exception_type, event.provider) == ("a", "failed", "OSError", "modal")


def test_collector_writes_pushed_events_to_disk(tmp_path: Path):
    collector = DashboardCollector(config=CollectorConfig(dashboard_dir=str(tmp_path), run_name="r", start_ts=0.0))
    collector.push_sandbox_events(EVENTS[:3])
    collector.flush()
    assert MetricStore.load(tmp_path).records[Stream.SANDBOX_EVENTS] == EVENTS[:3]


def test_report_converts_caller_values_to_json_safe_types(monkeypatch):
    handle = _FakeHandle()
    monkeypatch.setattr(backend, "_handle", handle)
    trial_id = uuid.uuid4()
    hooks.report_sandbox_event(trial_id, SandboxStage.VERIFIER_DONE, task=Path("t"), reward=np.float32(1.0))
    [([event],)] = handle.push_sandbox_events.calls
    assert (event.trial_id, event.task, event.reward) == (str(trial_id), "t", 1.0)
    assert type(event.reward) is float


def test_flush_drops_a_record_it_cannot_encode_and_keeps_the_rest(tmp_path: Path):
    collector = DashboardCollector(config=CollectorConfig(dashboard_dir=str(tmp_path), run_name="r", start_ts=0.0))
    bad = SandboxEvent(ts=9.0, trial_id="x", task="t", stage="created", reward=np.float32(1.0))
    collector.push_sandbox_events([EVENTS[0], bad, EVENTS[1]])
    collector.flush()
    assert MetricStore.load(tmp_path).records[Stream.SANDBOX_EVENTS] == EVENTS[:2]


def test_subproc_agent_events_reach_the_callers_collector(monkeypatch):
    handle = _FakeHandle()
    monkeypatch.setattr(backend, "_handle", handle)
    result = asyncio.run(
        call_agent_function(
            sandbox_event_stubs.report_sandbox_events,
            mode="subproc",
            on_child_reports=hooks.push_sandbox_events,
            trial_id="a",
        )
    )
    assert result == "reported"
    [(events,)] = handle.push_sandbox_events.calls
    assert [(event.trial_id, event.stage) for event in events] == [("a", "created"), ("a", "torn_down")]
