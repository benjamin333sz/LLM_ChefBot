import os
from dotenv import load_dotenv
from smolagents import LiteLLMModel

load_dotenv()

GROQ_OPENAI_BASE = "https://api.groq.com/openai/v1"


def get_groq_litellm_model(
    model_id: str = "groq/llama-3.3-70b-versatile",
    temperature: float = 0.2,
) -> LiteLLMModel:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY manquant dans .env / env vars.")

    return LiteLLMModel(
        model_id=model_id,
        api_base=GROQ_OPENAI_BASE,
        api_key=api_key,
        temperature=temperature,
    )
