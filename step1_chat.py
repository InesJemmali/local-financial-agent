import os 
import json
import requests


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:21434") # Default to localhost if not set

if not OLLAMA_HOST.startswith("http://"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST

MODEL = os.environ.get("MODEL", "qwen3:14b") # Default to qwen3:14b if not set   


def chat(messages,tools=None, think=False):
    """
    Send a chat request to the Ollama API.

    Args:
        messages (list): A list of message dictionaries.
        think (bool): Whether to enable thinking mode.

    Returns:
        dict: The response from the Ollama API.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "think": think,
        "stream": False,
        "options": {"temperature": 0},
    }

    if tools: 
        payload["tools"] = tools

    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json = payload,
        timeout = 600
    )

    response.raise_for_status()

    return response.json()

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a concise financial data analyst."},
        {"role": "user", "content": "In one sentence, what is transaction volume?"}
    ]
    raw = chat(messages)
    print("===== FULL RESPONSE JSON =====")
    print(json.dumps(raw, indent=2))

    print("\n===== MODEL RESPONSE =====")
    print(raw["message"]["content"])