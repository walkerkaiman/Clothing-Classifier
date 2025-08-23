from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

from pydantic import BaseModel, Field, field_validator, validator


class CameraConfig(BaseModel):
    id: int = Field(0, description="Index of the webcam device")
    width: int = Field(1280, ge=64, le=7680)
    height: int = Field(720, ge=64, le=4320)
    fps: int = Field(30, ge=1, le=240)


class ModelConfig(BaseModel):
    name: str = Field("yolov8m-cls", description="Registry key of the model to load")
    confidence: float = Field(0.25, ge=0, le=1)
    iou: float = Field(0.5, ge=0, le=1)


class AppConfig(BaseModel):
    data_dir: Path = Field(Path("data"))
    write_interval: float = Field(1.0, description="Seconds between state writes", ge=0.1, le=60)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field("INFO")

    camera: CameraConfig = Field(default_factory=CameraConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    @field_validator("data_dir")
    @classmethod
    def _ensure_dir(cls, v: Path):
        v.mkdir(parents=True, exist_ok=True)
        return v


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Settings":
        import tomli, json

        if path is None:
            return cls()

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix in {".toml", ".tml"}:
            data = tomli.loads(path.read_text())
        elif path.suffix in {".json"}:
            data = json.loads(path.read_text())
        else:
            raise ValueError(f"Unsupported config format: {path}")
        return cls.model_validate(data)
