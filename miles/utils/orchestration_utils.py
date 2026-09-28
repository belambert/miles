from dataclasses import dataclass
from functools import partial
from typing import Self

from miles.ray.wiring import compute_backend_capability, launch_worker_manager, shutdown_worker_manager
from miles.utils import object_store
from miles.utils.args.runtime import AllConfig, OrchestratorConfig
from miles.utils.arguments import parse_args
from miles.utils.async_utils import Disposer
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.logging_utils import configure_logger
from miles.utils.test_utils.fault_injector.actions.base import FaultHookResources
from miles.utils.test_utils.fault_injector.controller import fault_hook_controller
from miles.utils.test_utils.fault_injector.models import FaultHookOwner
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking
from miles.utils.workers.backend_capability.base import BackendCapability


@dataclass(frozen=True)
class ArgvOrchestratorStartupInfo:
    args: OrchestratorConfig
    all_args: AllConfig

    @classmethod
    def create(cls, all_args: AllConfig) -> Self:
        return cls(args=OrchestratorConfig.slice_from(all_args), all_args=all_args)


def parse_orchestrator_startup_info() -> ArgvOrchestratorStartupInfo:
    return ArgvOrchestratorStartupInfo.create(parse_args())


def init_orchestration_script(startup_info: ArgvOrchestratorStartupInfo, *, disposer: Disposer) -> BackendCapability:
    args = startup_info.args
    event_logger_checkpoint.restore(args)
    configure_logger(args, source=SimpleProcessIdentity(component="main"))
    maybe_start_periodic_pyspy_dump()
    disposer.add(finish_tracking)
    init_tracking(args)

    all_args = startup_info.all_args
    all_args.wandb_run_id = args.wandb_run_id
    all_args.mlflow_run_id = args.mlflow_run_id
    worker_manager = launch_worker_manager(all_args)
    disposer.add(partial(shutdown_worker_manager, worker_manager))

    capability = compute_backend_capability(all_args)
    object_store.init_instance(args, contribute_segment=False)
    fault_hook_controller.configure(resources=FaultHookResources(args=args), owner=FaultHookOwner.ORCHESTRATOR)
    return capability
