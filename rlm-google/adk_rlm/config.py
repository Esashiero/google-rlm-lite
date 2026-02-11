"""Configuration management for rlm-google."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class RLMConfig:
    """Configuration for RLM operations."""

    # Model settings
    default_model: str = "gemini/gemini-1.5-flash"

    # Rate limiting
    requests_per_minute: int = 60
    max_concurrent_requests: int = 30
    max_burst: int = 5

    # Retry settings
    max_retries: int = 5
    base_retry_delay: float = 1.0

    # API Keys (loaded from env vars)
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if self.gemini_api_key is None:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    @classmethod
    def from_env(cls) -> "RLMConfig":
        """Create config from environment variables."""
        return cls(
            default_model=os.getenv("RLM_DEFAULT_MODEL", "gemini/gemini-1.5-flash"),
            requests_per_minute=int(os.getenv("RLM_RPM", "60")),
            max_concurrent_requests=int(os.getenv("RLM_MAX_CONCURRENT", "30")),
            max_burst=int(os.getenv("RLM_MAX_BURST", "5")),
            max_retries=int(os.getenv("RLM_MAX_RETRIES", "5")),
            base_retry_delay=float(os.getenv("RLM_RETRY_DELAY", "1.0")),
        )


# Global config instance
_config: Optional[RLMConfig] = None


def get_config() -> RLMConfig:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = RLMConfig.from_env()
    return _config


def set_config(config: RLMConfig):
    """Set global config instance."""
    global _config
    _config = config
