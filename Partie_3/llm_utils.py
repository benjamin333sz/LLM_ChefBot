import json
from typing import Any, Dict, List

from dotenv import load_dotenv
from groq import Groq
from langfuse import get_client

load_dotenv()

groq_client = Groq()
langfuse = get_client()

MODEL_ID = "openai/gpt-oss-20b"


def chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    resp = groq_client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def safe_json_loads(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    # extraction JSON si le modèle ajoute du texte
    a, b = raw.find("{"), raw.rfind("}")
    if a != -1 and b != -1 and b > a:
        return json.loads(raw[a : b + 1])

    a, b = raw.find("["), raw.rfind("]")
    if a != -1 and b != -1 and b > a:
        return json.loads(raw[a : b + 1])

    raise ValueError("JSON introuvable dans la réponse.")
