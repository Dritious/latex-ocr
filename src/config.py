import yaml
from pydantic import BaseModel, SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class ModelConfig(BaseModel):
    d_model: int
    n_heads: int = Field(ge=1, description="Количество голов attention")
    max_seq_len: int

class TrainConfig(BaseModel):
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    lr: float = Field(gt=0.0)

class DatasetConfig(BaseModel):
    name: str
    source: str
    split: str = "train"
    columns_map: dict[str, str] = Field(default_factory=dict)

class AppSettings(BaseSettings):
    model: ModelConfig
    train: TrainConfig
    ru_datasets: list[DatasetConfig]
    latex_datasets: list[DatasetConfig]
    
    wandb_api_key: SecretStr = Field(...)
    wandb_project: str = "latex-ocr"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

def load_config(yaml_path: str = "params.yaml") -> AppSettings:
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    
    return AppSettings(**yaml_data)