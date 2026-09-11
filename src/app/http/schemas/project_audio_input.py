from pydantic import BaseModel, ConfigDict, Field


class ProjectAudioInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_id: str = Field(min_length=1, max_length=255)
