from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "PDA API"
    app_version: str = "0.1.0"

    model_config = SettingsConfigDict(env_prefix="PDA_", extra="ignore")


settings = Settings()
