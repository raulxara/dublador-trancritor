from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ManageVoiceDtoOut:
    unique_id: str
    name: str
    language_id: str
    gender_id: str | None
    current_sample_id: str | None
    status: str
    description: str | None
