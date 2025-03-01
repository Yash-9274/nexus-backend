from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Knowledge Management API"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str
    DIRECT_URL: str | None = None

    # JWT
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # OpenAI
    OPENAI_API_KEY: str | None = None
    COHERE_API_KEY: str | None = None

    # Document Storage
    DOCUMENT_STORAGE_PATH: str = "storage/documents"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {"pdf", "docx", "md"}

    # AWS Settings
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_BUCKET_NAME: str
    AWS_REGION: str
    
    # Update storage settings
    STORAGE_PROVIDER: str = "S3"

    class Config:
        env_file = ".env"

settings = Settings()