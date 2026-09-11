from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SubmitDubbingJobInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    operation: Literal["text_to_speech", "speech_to_speech", "transcribe"]
    input_text: str | None = Field(default=None, min_length=1, max_length=10000)
    input_message_id: str | None = Field(default=None, min_length=1, max_length=255)
    target_language_id: str | None = Field(default=None, min_length=1, max_length=255)
    speed: Decimal = Field(default=Decimal("1.00"), ge=Decimal("0.50"), le=Decimal("1.50"), decimal_places=2)
    pitch_semitones: Decimal = Field(default=Decimal("0.00"), ge=-6, le=6, decimal_places=2)
    preserve_timing: bool = False
