from dataclasses import dataclass

from app.entities.transcription.transcription_entity import TranscriptionEntity


@dataclass(frozen=True, slots=True)
class ListTranscriptionsDtoOut:
    data: tuple[TranscriptionEntity, ...]
