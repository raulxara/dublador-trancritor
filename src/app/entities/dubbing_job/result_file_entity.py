from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResultFileEntity:
    unique_id: str
    key: str
    format: str
    purpose: str
    size: int
    checksum: str
    duration_ms: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
