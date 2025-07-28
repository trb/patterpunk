from typing import Dict, List

from . import openai, anthropic, bedrock, google, ollama

PROVIDERS = {
    "openai": openai,
    "anthropic": anthropic,
    "bedrock": bedrock,
    "google": google,
    "ollama": ollama,
}

def get_available_providers() -> List[str]:
    available = []
    for name, provider in PROVIDERS.items():
        if hasattr(provider, 'is_available') and provider.is_available():
            available.append(name)
        elif hasattr(provider, f'is_{name}_available') and getattr(provider, f'is_{name}_available')():
            available.append(name)
    return available

def is_provider_available(provider_name: str) -> bool:
    if provider_name not in PROVIDERS:
        return False
    
    provider = PROVIDERS[provider_name]
    if hasattr(provider, 'is_available'):
        return provider.is_available()
    elif hasattr(provider, f'is_{provider_name}_available'):
        return getattr(provider, f'is_{provider_name}_available')()
    return False

def get_provider_module(provider_name: str):
    return PROVIDERS.get(provider_name)