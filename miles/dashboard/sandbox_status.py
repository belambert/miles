"""Read-side aggregation of the ``sandbox_events`` stream for the Sandboxes view."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from miles.dashboard.store import SandboxEvent, SandboxStage

TERMINAL_STAGES = frozenset({SandboxStage.TORN_DOWN, SandboxStage.FAILED})
UNKNOWN_EXCEPTION = "unknown"
MAX_FAILURE_BUCKETS = 2000


def sandbox_status(events: Sequence[SandboxEvent], *, now: float, stale_after_s: float, bucket_s: float) -> dict:
    """Stage counts from each trial's latest event, failure counts per
    exception type (total and per ``bucket_s`` time bucket, widened to keep at
    most ``MAX_FAILURE_BUCKETS``), and non-terminal trials silent for longer
    than ``stale_after_s`` before ``now``. A stale row keeps the trial's last
    known task, provider and sandbox id, which often only its first event has."""
    latest: dict[str, SandboxEvent] = {}
    labels: dict[str, dict[str, tuple[float, str]]] = {}
    for event in events:
        previous = latest.get(event.trial_id)
        if previous is None or event.ts >= previous.ts:
            latest[event.trial_id] = event
        known = labels.setdefault(event.trial_id, {})
        for field in ("task", "provider", "sandbox_id"):
            value = getattr(event, field)
            if value and (field not in known or event.ts >= known[field][0]):
                known[field] = (event.ts, value)

    stage_counts = {stage.value: 0 for stage in SandboxStage}
    stage_counts.update(Counter(event.stage for event in latest.values()))

    failures = [event for event in events if event.stage == SandboxStage.FAILED]
    failure_counts = Counter(event.exception_type or UNKNOWN_EXCEPTION for event in failures)
    failure_series = []
    if failures:
        t0 = min(event.ts for event in failures)
        span = max(event.ts for event in failures) - t0
        bucket_s = max(bucket_s, span / MAX_FAILURE_BUCKETS)
        num_buckets = min(int(span // bucket_s) + 1, MAX_FAILURE_BUCKETS + 1)
        for exception_type in sorted(failure_counts):
            counts = [0] * num_buckets
            for event in failures:
                if (event.exception_type or UNKNOWN_EXCEPTION) == exception_type:
                    counts[min(int((event.ts - t0) // bucket_s), num_buckets - 1)] += 1
            failure_series.append(
                dict(label=exception_type, ts=[t0 + i * bucket_s for i in range(num_buckets)], value=counts)
            )

    stale = sorted(
        (
            dict(
                trial_id=event.trial_id,
                task=_last_known(labels, event.trial_id, "task") or "",
                stage=event.stage,
                provider=_last_known(labels, event.trial_id, "provider"),
                sandbox_id=_last_known(labels, event.trial_id, "sandbox_id"),
                last_ts=event.ts,
                silent_s=now - event.ts,
            )
            for event in latest.values()
            if event.stage not in TERMINAL_STAGES and now - event.ts > stale_after_s
        ),
        key=lambda row: row["last_ts"],
    )
    return dict(
        num_trials=len(latest),
        stage_counts=stage_counts,
        failure_counts=dict(failure_counts.most_common()),
        failure_series=failure_series,
        stale=stale,
        now=now,
    )


def _last_known(labels: dict[str, dict[str, tuple[float, str]]], trial_id: str, field: str) -> str | None:
    known = labels[trial_id].get(field)
    return None if known is None else known[1]
