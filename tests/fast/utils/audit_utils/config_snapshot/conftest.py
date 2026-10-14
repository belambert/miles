from collections.abc import Callable

import pytest

from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotContext, ConfigSnapshotPoint, ConfigSnapshotRecord
from miles.utils.audit_utils.process_identity import TrainProcessIdentity


@pytest.fixture
def make_record() -> Callable[..., ConfigSnapshotRecord]:
    def create(
        *, run: int = 0, rank: int = 0, stage: str = "process_config", config: dict | None = None
    ) -> ConfigSnapshotRecord:
        return ConfigSnapshotRecord(
            context=ConfigSnapshotContext(
                name=f"test/run-{run:04d}",
                deploy_component="all",
                deploy_instance_id="default",
                source=TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=rank),
                run_uuid=f"uuid-{run}",
                capture_id=f"capture-{run}-{rank}",
            ),
            point=ConfigSnapshotPoint(stage=stage, index=0),
            config={"args": config if config is not None else {"rank": rank}},
        )

    return create
