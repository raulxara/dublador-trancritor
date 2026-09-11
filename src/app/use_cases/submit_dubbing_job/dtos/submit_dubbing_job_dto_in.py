import re
from dataclasses import dataclass
from decimal import Decimal

from app.exceptions.invalid_input_error import InvalidInputError
from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class SubmitDubbingJobDtoIn:
    actor: AuthorizedActor
    chat_id: str
    idempotency_key: str
    operation: str
    input_text: str | None = None
    input_message_id: str | None = None
    target_language_id: str | None = None
    speed: Decimal = Decimal("1.00")
    pitch_semitones: Decimal = Decimal("0.00")
    preserve_timing: bool = False

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,100}", self.idempotency_key):
            raise InvalidInputError()
        if self.operation not in ("text_to_speech", "speech_to_speech", "transcribe"):
            raise InvalidInputError()
        if (
            not self.speed.is_finite()
            or not self.pitch_semitones.is_finite()
            or not Decimal("0.50") <= self.speed <= Decimal("1.50")
            or not -6 <= self.pitch_semitones <= 6
            or self.speed != self.speed.quantize(Decimal("0.01"))
            or self.pitch_semitones != self.pitch_semitones.quantize(Decimal("0.01"))
        ):
            raise InvalidInputError()
        if self.operation == "text_to_speech":
            if (
                not self.input_text
                or not self.input_text.strip()
                or len(self.input_text) > 10000
                or self.input_message_id is not None
            ):
                raise InvalidInputError()
        elif self.input_text is not None or not self.input_message_id:
            raise InvalidInputError()
        if self.operation != "speech_to_speech" and self.preserve_timing:
            raise InvalidInputError()
        if self.operation == "transcribe" and (self.speed != 1 or self.pitch_semitones != 0):
            raise InvalidInputError()
