"""
Model Provider Configuration - Provider Agnostic Setup
Supports: OpenAI, Anthropic, Google, Azure OpenAI, Ollama, HuggingFace, and more
"""

import os
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


ProviderType = Literal[
    "openai",
    "azure_openai", 
    "anthropic",
    "google",
    "ollama",
    "huggingface",
    "bedrock",
    "cohere",
    "litellm"  # Unified interface for 100+ providers
]

EmbeddingProviderType = Literal[
    "openai",
    "azure_openai",
    "huggingface", 
    "ollama",
    "google",
    "cohere",
    "litellm"  # Unified interface for embeddings
]


@dataclass
class ModelConfig:
    """Configuration for LLM provider."""
    provider: ProviderType
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0
    max_tokens: Optional[int] = None
    additional_kwargs: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and set defaults based on provider."""
        if self.additional_kwargs is None:
            self.additional_kwargs = {}
            
        # Auto-detect API keys from environment
        if not self.api_key:
            self.api_key = self._get_api_key_from_env()
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment based on provider."""
        env_map = {
            "openai": "OPENAI_API_KEY",
            "azure_openai": "AZURE_OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "bedrock": "AWS_ACCESS_KEY_ID",
            "cohere": "COHERE_API_KEY",
        }
        env_var = env_map.get(self.provider)
        return os.getenv(env_var) if env_var else None


@dataclass
class EmbeddingConfig:
    """Configuration for embedding provider."""
    provider: EmbeddingProviderType
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    additional_kwargs: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.additional_kwargs is None:
            self.additional_kwargs = {}
            
        if not self.api_key:
            self.api_key = self._get_api_key_from_env()
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment."""
        env_map = {
            "openai": "OPENAI_API_KEY",
            "azure_openai": "AZURE_OPENAI_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "google": "GOOGLE_API_KEY",
            "cohere": "COHERE_API_KEY",
        }
        env_var = env_map.get(self.provider)
        return os.getenv(env_var) if env_var else None


def create_llm(config: ModelConfig):
    """
    Create LLM instance from configuration.
    
    Args:
        config: ModelConfig with provider and settings
        
    Returns:
        Initialized LangChain LLM instance
    """
    if config.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
            **config.additional_kwargs
        )
    
    elif config.provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            deployment_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
            azure_endpoint=config.base_url,
            **config.additional_kwargs
        )
    
    elif config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens or 4096,
            api_key=config.api_key,
            **config.additional_kwargs
        )
    
    elif config.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            google_api_key=config.api_key,
            **config.additional_kwargs
        )
    
    elif config.provider == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            model=config.model_name,
            base_url=config.base_url or "http://localhost:11434",
            temperature=config.temperature,
            **config.additional_kwargs
        )
    
    elif config.provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint
        return HuggingFaceEndpoint(
            repo_id=config.model_name,
            temperature=config.temperature,
            max_new_tokens=config.max_tokens,
            huggingfacehub_api_token=config.api_key,
            **config.additional_kwargs
        )
    
    elif config.provider == "bedrock":
        from langchain_aws import ChatBedrock
        return ChatBedrock(
            model_id=config.model_name,
            model_kwargs={
                "temperature": config.temperature,
                "max_tokens": config.max_tokens or 4096,
                **config.additional_kwargs
            }
        )
    
    elif config.provider == "cohere":
        from langchain_cohere import ChatCohere
        return ChatCohere(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            cohere_api_key=config.api_key,
            **config.additional_kwargs
        )
    
    elif config.provider == "litellm":
        # LiteLLM proxy exposes OpenAI-compatible API, so use ChatOpenAI with custom base_url
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=config.api_key,
            openai_api_base=config.base_url,
            **config.additional_kwargs
        )
    
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


def create_embeddings(config: EmbeddingConfig):
    """
    Create embeddings instance from configuration.
    
    Args:
        config: EmbeddingConfig with provider and settings
        
    Returns:
        Initialized LangChain Embeddings instance
    """
    if config.provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.model_name,
            api_key=config.api_key,
            **config.additional_kwargs
        )
    
    elif config.provider == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            model=config.model_name,
            api_key=config.api_key,
            azure_endpoint=config.base_url,
            **config.additional_kwargs
        )
    
    elif config.provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=config.model_name,
            **config.additional_kwargs
        )
    
    elif config.provider == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.model_name,
            base_url=config.base_url or "http://localhost:11434",
            **config.additional_kwargs
        )
    
    elif config.provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=config.model_name,
            google_api_key=config.api_key,
            **config.additional_kwargs
        )
    
    elif config.provider == "cohere":
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings(
            model=config.model_name,
            cohere_api_key=config.api_key,
            **config.additional_kwargs
        )
    
    elif config.provider == "litellm":
        from langchain_community.embeddings import LiteLLMEmbeddings
        # LiteLLM embeddings using provider/model format
        return LiteLLMEmbeddings(
            model=config.model_name,  # e.g., "openai/text-embedding-3-small"
            api_key=config.api_key,
            **config.additional_kwargs
        )
    
    else:
        raise ValueError(f"Unsupported embedding provider: {config.provider}")


# Preset configurations for common use cases
PRESET_CONFIGS = {
    "openai-gpt4": ModelConfig(
        provider="openai",
        model_name="gpt-4o"
    ),
    "openai-gpt4-turbo": ModelConfig(
        provider="openai",
        model_name="gpt-4-turbo-preview"
    ),
    "openai-gpt35": ModelConfig(
        provider="openai",
        model_name="gpt-3.5-turbo"
    ),
    "anthropic-claude": ModelConfig(
        provider="anthropic",
        model_name="claude-3-5-sonnet-20241022"
    ),
    "anthropic-haiku": ModelConfig(
        provider="anthropic",
        model_name="claude-3-5-haiku-20241022"
    ),
    "google-gemini": ModelConfig(
        provider="google",
        model_name="gemini-1.5-pro"
    ),
    "google-gemini-flash": ModelConfig(
        provider="google",
        model_name="gemini-1.5-flash"
    ),
    "ollama-llama3": ModelConfig(
        provider="ollama",
        model_name="llama3.1:8b"
    ),
    "ollama-mistral": ModelConfig(
        provider="ollama",
        model_name="mistral:7b"
    ),
    # LiteLLM - unified interface for 100+ providers
    "litellm-gpt4": ModelConfig(
        provider="litellm",
        model_name="openai/gpt-4o"
    ),
    "litellm-claude": ModelConfig(
        provider="litellm",
        model_name="anthropic/claude-3-5-sonnet-20241022"
    ),
    "litellm-gemini": ModelConfig(
        provider="litellm",
        model_name="gemini/gemini-1.5-pro"
    ),
}

PRESET_EMBEDDINGS = {
    "openai": EmbeddingConfig(
        provider="openai",
        model_name="text-embedding-3-small"
    ),
    "openai-large": EmbeddingConfig(
        provider="openai",
        model_name="text-embedding-3-large"
    ),
    "huggingface": EmbeddingConfig(
        provider="huggingface",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ),
    "ollama": EmbeddingConfig(
        provider="ollama",
        model_name="nomic-embed-text"
    ),
    "google": EmbeddingConfig(
        provider="google",
        model_name="models/embedding-001"
    ),
    # LiteLLM embeddings
    "litellm": EmbeddingConfig(
        provider="litellm",
        model_name="openai/text-embedding-3-small"
    ),
}


def get_preset_config(preset_name: str) -> ModelConfig:
    """Get a preset model configuration."""
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[preset_name]


def get_preset_embeddings(preset_name: str) -> EmbeddingConfig:
    """Get a preset embedding configuration."""
    if preset_name not in PRESET_EMBEDDINGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_EMBEDDINGS.keys())}")
    return PRESET_EMBEDDINGS[preset_name]
