from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class JobOutputEntity:
    unique_id: str
    format: str
    purpose: str
    size_bytes: int
    duration_ms: int | None
    storage_key: str = field(repr=False)
