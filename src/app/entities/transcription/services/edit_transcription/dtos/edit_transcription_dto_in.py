from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EditTranscriptionDtoIn:
    office_id: str
    user_id: str
    owner_id: str
    job_id: str
    base_transcription_id: str
    text: str
