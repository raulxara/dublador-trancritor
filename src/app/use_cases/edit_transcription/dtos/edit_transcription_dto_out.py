from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EditTranscriptionDtoOut:
    data: dict
