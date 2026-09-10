from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FindVoiceByUniqueIdDtoIn:
    office_id: str
    unique_id: str

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v.strip() for v in (self.office_id, self.unique_id)):
            raise ValueError("office_id and unique_id are required")
