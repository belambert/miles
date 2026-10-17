"""An agent function that reports sandbox events, kept apart from ``agent_function_stubs``
so the other ``subproc`` tests do not import the dashboard in every child process."""

from miles.dashboard.hooks import report_sandbox_event
from miles.dashboard.store import SandboxStage


async def report_sandbox_events(trial_id: str, **kwargs) -> str:
    report_sandbox_event(trial_id, SandboxStage.CREATED, provider="local")
    report_sandbox_event(trial_id, SandboxStage.TORN_DOWN)
    return "reported"
