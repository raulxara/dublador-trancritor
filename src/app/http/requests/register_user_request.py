import re

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class RegisterUserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    officeId: str | None = Field(default=None, min_length=1, max_length=255)
    firstName: str = Field(min_length=1, max_length=255)
    lastName: str = Field(min_length=1, max_length=255)
    email: str = Field(min_length=3, max_length=255)
    phone: str = Field(min_length=8, max_length=40)
    username: str = Field(min_length=1, max_length=255)
    password: SecretStr = Field(min_length=12, max_length=1024)
    documentType: str | None = Field(default=None, min_length=1, max_length=30)
    documentValue: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("email")
    @classmethod
    def email_format(cls, value: str) -> str:
        value = value.lower()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("Invalid email")
        return value

    @field_validator("phone")
    @classmethod
    def phone_format(cls, value: str) -> str:
        if not re.fullmatch(r"\+?[1-9][0-9]{7,14}", value):
            raise ValueError("Use international phone format")
        return value
