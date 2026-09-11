from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DUBBER_WORKER_", extra="ignore")
    model_root: str = "/var/cache/dubber/models"
    lease_seconds: int = Field(default=120, ge=30)
    heartbeat_seconds: int = Field(default=10, ge=1)
    timeout_seconds: int = Field(default=1200, ge=10)
    max_attempts: int = Field(default=3, ge=1, le=10)
    poll_seconds: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def validate_policy(self):
        if self.heartbeat_seconds * 3 >= self.lease_seconds:
            raise ValueError("Lease must exceed three heartbeat intervals")
        return self
