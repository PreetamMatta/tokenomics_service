"""Runtime configuration: config/settings.yaml + environment overrides.

Environment variables:
- CONFIG_DIR: config directory (default ./config)
- OPENROUTER_TTL_SECONDS / AA_TTL_SECONDS: cache TTL overrides
- AA_API_KEY (name configurable via settings.yaml artificial_analysis.api_key_env):
  optional — when absent the service degrades gracefully to no derived tiers.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .benchmarks.artificial_analysis import DEFAULT_AA_URL
from .providers.openrouter import DEFAULT_OPENROUTER_URL


@dataclass
class Settings:
    config_dir: Path
    openrouter_url: str
    openrouter_ttl_seconds: float
    aa_url: str
    aa_ttl_seconds: float
    aa_api_key: str | None
    tiers_path: Path
    aliases_path: Path
    overrides_path: Path
    local_models_dir: Path


def load_settings(config_dir: str | Path | None = None) -> Settings:
    directory = Path(config_dir or os.environ.get("CONFIG_DIR", "config"))
    raw: dict = {}
    settings_file = directory / "settings.yaml"
    if settings_file.is_file():
        raw = yaml.safe_load(settings_file.read_text()) or {}

    openrouter = raw.get("openrouter") or {}
    aa = raw.get("artificial_analysis") or {}
    files = raw.get("files") or {}

    openrouter_ttl = float(
        os.environ.get("OPENROUTER_TTL_SECONDS", openrouter.get("ttl_seconds", 86_400))
    )
    aa_ttl = float(os.environ.get("AA_TTL_SECONDS", aa.get("ttl_seconds", 86_400)))
    api_key_env = aa.get("api_key_env", "AA_API_KEY")

    return Settings(
        config_dir=directory,
        openrouter_url=openrouter.get("url", DEFAULT_OPENROUTER_URL),
        openrouter_ttl_seconds=openrouter_ttl,
        aa_url=aa.get("url", DEFAULT_AA_URL),
        aa_ttl_seconds=aa_ttl,
        aa_api_key=os.environ.get(api_key_env) or None,
        tiers_path=directory / files.get("tiers", "tiers.yaml"),
        aliases_path=directory / files.get("aliases", "model_aliases.yaml"),
        overrides_path=directory / files.get("overrides", "overrides.yaml"),
        local_models_dir=directory / files.get("local_models_dir", "local_models"),
    )
