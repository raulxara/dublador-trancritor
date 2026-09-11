from pydantic import BaseModel, ConfigDict, Field


class VoiceTagsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tag_ids: list[str] = Field(max_length=50)
