from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FindVoiceByUniqueIdDtoOut:
    unique_id: str
    office_id: str
    name: str
    language_id: str
    gender_id: str | None
    current_sample_id: str | None
    status: str
