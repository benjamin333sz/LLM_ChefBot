from LLM_judge import create_chefbot_dataset,run_experiment
from langfuse import get_client,observe,propagate_attributes
from llm_utils import langfuse
GROUP = "Groupe_SZUREK_KUSNIEREK_GOSSELIN"

@observe(name=f"{GROUP}_Partie_3",as_type="chain")
def main():
    get_client().update_current_span(metadata={"partie": "3", "status": "start"})
    
    with propagate_attributes(tags=["Partie_3", GROUP]):
        print("=" * 60)
        print("CHEFBOT - DATASET + EVALUATION + EXPERIMENT")
        print("=" * 60)

        create_chefbot_dataset()
        run_experiment()

        get_client().update_current_span(metadata={"status": "success"})
        langfuse.flush()
        print("✓ Flushed to Langfuse")


if __name__ == "__main__":
    main()