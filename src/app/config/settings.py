from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DUBBER_", extra="ignore")
    app_name: str = Field(default="SiPlug Dubber API", min_length=1)
    environment: str = "local"
    db_host: str = "db"
    db_port: int = Field(default=3306, ge=1, le=65535)
    db_database: str = "siplug_dubber"
    db_username: str = "siplug_dubber"
    db_password: SecretStr = Field(min_length=1)
