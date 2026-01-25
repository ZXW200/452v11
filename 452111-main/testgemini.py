"""
Gemini API 测试脚本 - 显示完整错误信息
"""
import requests

API_KEY = "AIzaSyCpu6EJIwv6yCFO63Abu7hWy59QslfmNU0"

# 试几个不同的模型
models = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-3-pro-preview",
]

for model in models:
    print(f"\n测试模型: {model}")
    print("-" * 40)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"

    data = {
        "contents": [{"role": "user", "parts": [{"text": "Say OK"}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 10,
        }
    }

    try:
        resp = requests.post(url, json=data, timeout=30)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")