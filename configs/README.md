# Configs: centralized configuration and LLM profiles

This folder stores example configuration files and recommended conventions for centralizing model and environment configuration for the repository.

Conventions
- `llm_profiles.yaml` - a set of named LLM profiles (provider, model, optional api_key references). Use these profiles from code to select model providers at runtime.
- Use environment variables for secrets. See `.env.example` in the repository root.
- Profiles should avoid hardcoding secrets. If an API key is required, prefer storing it in the environment and reference it from the profile (e.g., `env: OPENAI_API_KEY`).

Example usage

```py
from utils.llm_manager import LLMManager

mgr = LLMManager(config_path='configs/llm_profiles.yaml')
client = mgr.get_client('openai-default')
print(client.generate('Summarize the repo in one sentence'))
```

Add new profiles when creating example apps under `examples/` so the examples can refer to named profiles instead of hardcoding providers.
