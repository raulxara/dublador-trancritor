from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranscriptionEntity:
    unique_id: str
    office_id: str
    job_id: str
    source_file_id: str
    language_id: str | None
    text: str
    version: int
    origin: str
    edited_by_user_id: str | None
    previous_transcription_id: str | None
    status: str = "active"
