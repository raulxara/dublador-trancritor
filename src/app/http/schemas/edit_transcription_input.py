from pydantic import BaseModel, ConfigDict, Field


class EditTranscriptionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_transcription_id: str = Field(min_length=1, max_length=255)
    text: str = Field(min_length=1, max_length=10000)
