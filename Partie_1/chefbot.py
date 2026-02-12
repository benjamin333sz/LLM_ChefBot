from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client, propagate_attributes
import json

load_dotenv()

groq_client = Groq()
langfuse = get_client()

@observe(name="Groupe_SZUREK_KUSNIEREK_GOSSELIN_Partie_1")
def ask_chef(question:str,temperature:float)->str:
    with propagate_attributes(tags=["Partie_1","Groupe_SZUREK_KUSNIEREK_GOSSELIN"]):
        try:
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
                metadata={
                        "type": "response",
                        "season":"winter",
                        "output":response.choices[0].message.content,
                        "temperature":temperature,
                        "partie":"1",
                        "status": "success"
                        }
            )
        except Exception as e:
            get_client().update_current_span(
                level="ERROR",
                status_message=str(e)
            )
    return response.choices[0].message.content

if __name__ == "__main__":
    prompt = """
    Que proposez-vous comme repas pour ce midi ?
    """
    temperatures = [0.1, 0.7, 1.2]
    for temperature in temperatures:
        print(
            f"Temperature at {temperature}\n\n",
            "_" * 50,
            ask_chef(question=prompt, temperature=temperature),
            end="\n\n",
        )
        # Pour une température de 0.1 :
        # Le retour est très court et possède moins de détail. 

        # Pour une température de 0.7 :
        # Le retour est un peu plus long et un plus détaillé

        # Pour une température de 1.2 :
        # Le retour est bien plus long que les deux précédents essaies,
        # avec beaucoup de précision et même l'avis du modèle.

        # La température du modèle fait varier la "créativité" du modèle.
        # Une température basse prendra des réponses très stricts (prend
        # les probabilités les plus hautes et c'est tout).
        # Une température élevé aura des réponses bien plus flexible (probabilité plus réparti)
        

    langfuse.flush()
