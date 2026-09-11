from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RegisterVoiceSampleDtoOut:
    unique_id: str
    voice_id: str
    version: int
    validation_state: str
    status: str
