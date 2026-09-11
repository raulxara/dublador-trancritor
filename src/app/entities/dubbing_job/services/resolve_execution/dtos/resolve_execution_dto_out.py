from dataclasses import dataclass

from app.entities.dubbing_job.execution_input_entity import ExecutionInputEntity


@dataclass(frozen=True, slots=True)
class ResolveExecutionDtoOut:
    data: ExecutionInputEntity | None
