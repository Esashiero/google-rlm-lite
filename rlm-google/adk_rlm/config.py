"""Configuration management for rlm-google."""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider."""
    requests_per_minute: int = 60
    max_concurrent: int = 30
    max_burst: int = 5


@dataclass
class RLMConfig:
    """Configuration for RLM operations."""

    # Model settings
    default_model: str = "gemini/gemini-1.5-flash"
    max_iterations: int = 30

    # Default Rate limiting (if not specified per provider)
    requests_per_minute: int = 60
    max_concurrent_requests: int = 30
    max_burst: int = 5

    # Provider-specific overrides
    # Example env var: RLM_PROVIDER_LIMITS='{"mistral": {"requests_per_minute": 30, "max_burst": 1}}'
    provider_configs: Dict[str, ProviderConfig] = field(default_factory=lambda: {
        "mistral": ProviderConfig(requests_per_minute=20, max_burst=1, max_concurrent=1),
        "gemini": ProviderConfig(requests_per_minute=60, max_burst=5, max_concurrent=30),
    })

    # Retry settings
    max_retries: int = 5
    base_retry_delay: float = 1.0

    # API Keys (loaded from env vars)
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None

    def __post_init__(self):
        """Load API keys and provider configs from environment if not provided."""
        if self.gemini_api_key is None:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.mistral_api_key is None:
            self.mistral_api_key = os.getenv("MISTRAL_API_KEY")

        # Load provider overrides from JSON env var if present
        overrides_json = os.getenv("RLM_PROVIDER_LIMITS")
        if overrides_json:
            try:
                overrides = json.loads(overrides_json)
                for provider, limits in overrides.items():
                    if provider not in self.provider_configs:
                        self.provider_configs[provider] = ProviderConfig()

                    for key, value in limits.items():
                        if hasattr(self.provider_configs[provider], key):
                            setattr(self.provider_configs[provider], key, value)
            except Exception as e:
                print(f"Error parsing RLM_PROVIDER_LIMITS: {e}")

    def get_provider_config(self, provider: str) -> ProviderConfig:
        """Get config for a specific provider, falling back to defaults."""
        if provider in self.provider_configs:
            config = self.provider_configs[provider]
        else:
            config = ProviderConfig(
                requests_per_minute=self.requests_per_minute,
                max_concurrent=self.max_concurrent_requests,
                max_burst=self.max_burst
            )

        logger.info(f"Using config for {provider}: {config}")
        return config

    @classmethod
    def from_env(cls) -> "RLMConfig":
        """Create config from environment variables."""
        return cls(
            default_model=os.getenv("RLM_MODEL", os.getenv("RLM_DEFAULT_MODEL", "gemini/gemini-1.5-flash")),
            max_iterations=int(os.getenv("RLM_MAX_ITERATIONS", "30")),
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
