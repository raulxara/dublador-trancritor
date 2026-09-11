from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VoiceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=255)
    language_id: str = Field(min_length=1, max_length=255)
    gender_id: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=10000)


class VoiceUpdateInput(VoiceInput):
    status: Literal["active", "inactive"] = "active"
