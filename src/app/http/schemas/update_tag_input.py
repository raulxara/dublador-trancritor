from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class UpdateTagInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=255)
    slug: str = Field(min_length=1, max_length=255, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    status: Literal["active", "inactive"] = "active"
