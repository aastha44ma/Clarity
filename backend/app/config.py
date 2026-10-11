from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    app_name: str = "AI Document Analyzer"
    environment: str = "development"
    frontend_origins: str = "https://clarity-umber-eta.vercel.app"

    # Dynamically pull from environment variables first, fallback to empty string
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "doc-analyzer")

    model_config = SettingsConfigDict(extra="ignore")

    @property
    def allowed_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.frontend_origins.split(",")
            if origin.strip()
        ]

settings = Settings()
