from pydantic import BaseModel, ConfigDict, Field


class CreateDubbingChatInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str | None = Field(default=None, min_length=1, max_length=255)
    selected_voice_id: str | None = Field(default=None, min_length=1, max_length=255)
