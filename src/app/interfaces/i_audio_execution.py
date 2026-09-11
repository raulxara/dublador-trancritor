from typing import Callable, Protocol

from app.entities.dubbing_job.execution_input_entity import ExecutionInputEntity
from app.entities.dubbing_job.execution_result_entity import ExecutionResultEntity


class IAudioExecution(Protocol):
    def run(self, context: ExecutionInputEntity, renew: Callable[[], bool]) -> ExecutionResultEntity: ...
