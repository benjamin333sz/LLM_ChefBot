from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client, propagate_attributes
import json

load_dotenv()

groq_client = Groq()
langfuse = get_client()

@observe(name="Groupe_SZUREK_KUSNIEREK_GOSSELIN")
def ask_chef(question:str,temperature:float)->str:
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
        temperature=temperature
        )


    get_client().update_current_span(
        metadata={"type": "response",
                  "season":"winter",
                  "output":response.choices[0].message.content,
                  "temperature":temperature,
                  "partie":"1",
                  }
    )

    return response.choices[0].message.content

prompt="""
Que proposez-vous comme repas pour ce midi ?
"""
temperatures=[0.1,0.7,1.2]
for temperature in temperatures:
    print(f"Temperature at {temperature}\n\n",
          "_"*50,
          ask_chef(question=prompt,temperature=temperature),
          end="\n\n")
    

langfuse.flush()