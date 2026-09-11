from dataclasses import dataclass
from decimal import Decimal

from app.exceptions.invalid_input_error import InvalidInputError


@dataclass(frozen=True, slots=True)
class DubbingJobEntity:
    unique_id: str
    office_id: str
    user_customer_id: str
    chat_id: str
    input_message_id: str
    operation: str
    input_text: str | None
    input_file_id: str | None
    voice_sample_id: str | None
    target_language_id: str | None
    speed: Decimal
    pitch_semitones: Decimal
    preserve_timing: bool
    parameters: str
    idempotency_key: str
    request_hash: str
    processing_state: str = "queued"
    status: str = "active"

    def __post_init__(self):
        if (
            not self.unique_id
            or not self.office_id
            or not self.chat_id
            or self.processing_state not in ("queued", "processing", "completed", "failed", "cancelled")
            or self.operation not in ("text_to_speech", "speech_to_speech", "transcribe")
            or self.status not in ("active", "inactive")
        ):
            raise InvalidInputError()

    output_message_id: str | None = None
    error_code: str | None = None
    attempts: int = 0
