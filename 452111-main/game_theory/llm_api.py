"""
Unified LLM API client. Supports DeepSeek, OpenAI, Gemini, Ollama.
Uses shared requests.Session for connection pooling.
"""

import os
import json
from typing import Optional, Dict, List

import requests

# Shared session for TCP connection reuse (thread-safe)
_shared_session: requests.Session = None


def get_shared_session() -> requests.Session:
    """Get or create the shared requests.Session singleton."""
    global _shared_session
    if _shared_session is None:
        _shared_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=100,
            pool_maxsize=100,
            max_retries=3
        )
        _shared_session.mount('http://', adapter)
        _shared_session.mount('https://', adapter)
    return _shared_session

# Config file path
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "llm_config.json")

DEFAULT_CONFIG = {
    "default_provider": "deepseek",
    
    "openai": {
        "api_key": "",
        "base_url": "https://hiapi.online/v1",
        "model": "gpt-4o-mini",
    },
    "gemini": {
        "api_key": "",
        "base_url": "https://generativelanguage.googleapis.com",
        "model": "gemini-2.0-flash",
    },
    "deepseek": {
        "api_key": "",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },

}


def load_config() -> Dict:
    """Load config from file, create default if missing."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG


def save_config(config: Dict):
    """Save config to JSON file."""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_api_key(provider: str) -> str:
    """Get API key. Environment variable takes priority over config file."""
    env_keys = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    if provider in env_keys:
        env_val = os.environ.get(env_keys[provider])
        if env_val:
            return env_val
    config = load_config()
    return config.get(provider, {}).get("api_key", "")


class LLMClient:
    """Unified LLM client with connection pooling."""

    def __init__(self, provider: str = None, session: requests.Session = None):
        self.config = load_config()
        provider = provider or self.config.get("default_provider", "deepseek")
        self.provider = provider
        self.session = session or get_shared_session()
        
    def chat(self,
             prompt: str,
             system_prompt: str = None,
             temperature: float = 0.7,
             max_tokens: int = 500) -> str:
        """Send a chat request and return the response text."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        if self.provider == "gemini":
            return self._call_gemini(messages, temperature, max_tokens)
        elif self.provider == "ollama":
            return self._call_ollama(messages, temperature, max_tokens)
        else:
            return self._call_openai_compatible(messages, temperature, max_tokens)
    
    def _call_openai_compatible(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """Call OpenAI-compatible API (OpenAI/DeepSeek)."""
        provider_config = self.config.get(self.provider, {})
        api_key = get_api_key(self.provider)
        base_url = provider_config.get("base_url", "https://api.openai.com/v1")
        model = provider_config.get("model", "gpt-4o-mini")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            resp = self.session.post(f"{base_url}/chat/completions", headers=headers, json=data, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[API Error: {e}]"
    
    def _call_gemini(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """Call Gemini native REST API."""
        api_key = get_api_key("gemini")
        model = self.config.get("gemini", {}).get("model", "gemini-1.5-flash")

        # Convert message format to Gemini's format
        contents = []
        system_instruction = None
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        if system_instruction:
            data["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        try:
            resp = self.session.post(url, json=data, timeout=60)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"[Gemini Error: {e}]"
    
    def _call_ollama(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """Call local Ollama instance."""
        ollama_config = self.config.get("ollama", {})
        base_url = ollama_config.get("base_url", "http://localhost:11434")
        model = ollama_config.get("model", "llama3")

        data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }

        try:
            resp = self.session.post(f"{base_url}/api/chat", json=data, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            return f"[Ollama Error: {e}]"
    
    def test_connection(self) -> bool:
        """Test if API connection works."""
        try:
            response = self.chat("Say OK", max_tokens=10)
            return not response.startswith("[") and len(response) > 0
        except Exception:
            return False


def chat(prompt: str, provider: str = None, **kwargs) -> str:
    """Quick chat shortcut."""
    client = LLMClient(provider=provider)
    return client.chat(prompt, **kwargs)


def setup_wizard():
    """Interactive setup wizard for API keys."""
    print("\n" + "="*50)
    print("LLM API Configuration")
    print("="*50)
    
    config = load_config()
    
    print("\n可用 Provider:")
    print("  1. deepseek  - ¥1/百万token，最便宜 (推荐)")
    print("  2. openai    - GPT-4o-mini")
    print("  3. gemini    - Gemini 1.5 Flash")
    print("  4. ollama    - 本地模型，免费")

    choice = input("\n选择 (1-4) [1]: ").strip() or "1"
    provider = {"1": "deepseek", "2": "openai", "3": "gemini", "4": "ollama"}.get(choice, "deepseek")
    config["default_provider"] = provider
    
    if provider != "ollama":
        api_key = input(f"\n输入 {provider.upper()} API Key: ").strip()
        if api_key:
            config[provider]["api_key"] = api_key
    
    save_config(config)
    print(f"\nConfig saved")
    
    print("\n测试连接...")
    client = LLMClient(provider=provider)
    if client.test_connection():
        print("Connection OK")
    else:
        print("Connection failed, check API Key")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_wizard()
    else:
        print("运行 python llm_api.py setup 进行配置")
