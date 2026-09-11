from dataclasses import dataclass

from app.entities.dubbing_job.result_file_entity import ResultFileEntity
from app.entities.dubbing_job.segment_entity import SegmentEntity


@dataclass(frozen=True, slots=True)
class ExecutionResultEntity:
    files: tuple[ResultFileEntity, ...]
    text: str | None
    language: str | None
    segments: tuple[SegmentEntity, ...]
    engine: str
    model_version: str
