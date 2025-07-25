from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    GOOGLE_API_KEY: str | None = None
    DATABASE_URL: str = "sqlite:///./college.db"
    
    class Config:
        env_file = ".env"

settings = Settings()