from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SegmentEntity:
    start_ms: int
    end_ms: int
    text: str
