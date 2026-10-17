"""An agent function that reports its sandbox, kept apart from ``agent_function_stubs``
so the other ``subproc`` tests do not import the dashboard in every child process."""

from miles.dashboard.hooks import report_sandbox


async def report_sandbox_health(sandbox_key: str, **kwargs) -> str:
    report_sandbox(sandbox_key, provider="local")
    report_sandbox(sandbox_key, status="finished")
    return "reported"
