from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionInputEntity:
    office_id: str
    user_id: str
    chat_id: str
    message_id: str
    operation: str
    input_text: str | None
    input_key: str | None
    sample_key: str | None
    language: str | None
    speed: float
    pitch_semitones: float
    preserve_timing: bool
