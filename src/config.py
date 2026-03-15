from pathlib import Path

from pydantic import Field

try:
    # Pydantic v2+
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _IS_PYDANTIC_V2 = True
except ImportError:
    # Pydantic v1 fallback
    from pydantic import BaseSettings
    _IS_PYDANTIC_V2 = False

Base_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    app_env: str = Field(default='development', description='Application environment, e.g. development/production')
    data_dir: Path = Field(default=Base_DIR / 'data', description='Directory where the dataset is stored')
    model_dir: Path = Field(default=Base_DIR / 'models', description='Directory where trained models are saved')
    random_seed: int = Field(default=42, description='Random seed for reproducibility')

    if _IS_PYDANTIC_V2:
        model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')
    else:
        class Config:
            env_file = '.env'
            env_file_encoding = 'utf-8'

settings = Settings()