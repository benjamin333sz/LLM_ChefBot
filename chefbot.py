from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client, propagate_attributes
import json

load_dotenv()

groq_client = Groq()
langfuse = get_client()

@observe()
def ask_chef(question:str)->str:
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": 
                """
                Tu es un chef cuisinier français spécialisé en cuisine de saison
                """
            },
            {"role": "user", "content": question}
        ],
        temperature=0.2
        )

    plan = json.loads(response.choices[0].message.content)

    get_client().update_current_span(
        metadata={"num_steps": len(plan.get("steps", []))}
    )

    return


get_client.flush()