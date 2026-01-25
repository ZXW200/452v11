"""
OpenAI API 测试脚本 - 显示完整错误信息
"""
import requests

API_KEY = "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"

url = "https://api.openai.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

data = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 10
}

print("测试 OpenAI API...")
print(f"URL: {url}")
print(f"Model: {data['model']}")
print("-" * 50)

try:
    resp = requests.post(url, headers=headers, json=data, timeout=30)

    print(f"Status Code: {resp.status_code}")
    print(f"Response Headers:")
    for k, v in resp.headers.items():
        if k.lower() in ['x-request-id', 'cf-ray', 'x-ratelimit-remaining-requests']:
            print(f"  {k}: {v}")

    print("-" * 50)
    print("Response Body:")
    print(resp.text)

except Exception as e:
    print(f"Error: {e}")