from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class VoiceEntity:
    unique_id: str  # voices._id, never the numeric id
    office_id: str
    name: str
    language_id: str
    gender_id: str | None = None
    current_sample_id: str | None = None
    status: Literal["active", "inactive"] = "active"

    def __post_init__(self) -> None:
        if any(
            not isinstance(v, str) or not v.strip()
            for v in (self.unique_id, self.office_id, self.name, self.language_id)
        ):
            raise ValueError("Voice identifiers and name are required")
        if self.status not in ("active", "inactive"):
            raise ValueError("Invalid voice status")
