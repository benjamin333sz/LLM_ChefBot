from __future__ import annotations

import json
from dotenv import load_dotenv
from smolagents import tool, CodeAgent, LiteLLMModel

load_dotenv()

# Adapte le model_id selon ton setup LiteLLM
# Exemple Groq:
model = LiteLLMModel(model_id="groq/llama-3.3-70b-versatile")

# =============================================================================
# 4.3 - TOOLS WITH @tool (docstrings include param descriptions!)
# =============================================================================

@tool
def check_fridge() -> str:
    """
    Return the list of ingredients currently available in the fridge (simulated).

    Returns:
        A JSON string containing a list under the key "available".
    """
    data = {
        "available": [
            "tomates", "oignons", "ail", "riz", "lentilles", "pois chiches", "tofu",
            "œufs", "yaourt", "citron", "huile d'olive", "basilic", "origan",
            "poivrons", "épinards",
        ]
    }
    return json.dumps(data, ensure_ascii=False)


@tool
def get_recipe(dish_name: str) -> str:
    """
    Return a detailed recipe for a given dish name (simulated).

    Args:
        dish_name: Name of the dish to retrieve (e.g., "curry de pois chiches", "shakshuka").

    Returns:
        A JSON string with fields like "title", "ingredients", "steps", and "time_minutes".
    """
    recipes = {
        "bol de lentilles citronnées": {
            "title": "Bol de lentilles citronnées",
            "ingredients": ["lentilles", "oignons", "ail", "citron", "huile d'olive", "épinards", "épices"],
            "steps": [
                "Cuire les lentilles (ou utiliser des lentilles déjà cuites).",
                "Sauter oignons + ail, ajouter lentilles + épices.",
                "Ajouter épinards, finir avec jus + zeste de citron.",
            ],
            "time_minutes": 20,
        },
        "curry de pois chiches": {
            "title": "Curry de pois chiches facile",
            "ingredients": ["pois chiches", "tomates", "oignons", "ail", "épinards", "épices", "huile d'olive"],
            "steps": [
                "Faire revenir oignons + ail + épices.",
                "Ajouter tomates + pois chiches, mijoter 12 min.",
                "Ajouter épinards 2 min, ajuster l'assaisonnement.",
            ],
            "time_minutes": 20,
        },
        "shakshuka": {
            "title": "Shakshuka express",
            "ingredients": ["tomates", "oignons", "ail", "poivrons", "œufs", "huile d'olive", "épices"],
            "steps": [
                "Faire revenir oignons + poivrons dans l'huile d'olive.",
                "Ajouter ail + tomates, mijoter 10 min.",
                "Former 2-3 puits, casser les œufs, couvrir 5-7 min.",
                "Servir avec herbes fraîches.",
            ],
            "time_minutes": 25,
        },
    }

    key = dish_name.strip().lower()
    recipe = recipes.get(
        key,
        {"title": dish_name.strip(), "ingredients": [], "steps": ["Recette non disponible (simulé)."], "time_minutes": None},
    )
    return json.dumps(recipe, ensure_ascii=False)


@tool
def check_dietary_info(ingredient: str) -> str:
    """
    Return nutritional + allergen info for a given ingredient (simulated).

    Args:
        ingredient: Ingredient name to analyze (e.g., "arachide", "œufs", "tofu").

    Returns:
        A JSON string with keys: "ingredient", "allergens", "vegan", and "notes".
    """
    db = {
        "arachide": {"allergens": ["arachide"], "vegan": True, "notes": "Allergène majeur."},
        "cacahuète": {"allergens": ["arachide"], "vegan": True, "notes": "Allergène majeur."},
        "œufs": {"allergens": ["œuf"], "vegan": False, "notes": "Produit animal."},
        "lait": {"allergens": ["lait"], "vegan": False, "notes": "Lactose possible."},
        "yaourt": {"allergens": ["lait"], "vegan": False, "notes": "Produit laitier."},
        "tofu": {"allergens": ["soja"], "vegan": True, "notes": "Source de protéines végétales."},
        "lentilles": {"allergens": [], "vegan": True, "notes": "Riche en fibres et protéines."},
        "pois chiches": {"allergens": [], "vegan": True, "notes": "Bon pour budget + satiété."},
        "riz": {"allergens": [], "vegan": True, "notes": "Naturellement sans gluten."},
    }

    key = ingredient.strip().lower()
    info = db.get(key, {"allergens": [], "vegan": None, "notes": "Info non disponible (simulé)."})
    return json.dumps({"ingredient": ingredient, **info}, ensure_ascii=False)


# =============================================================================
# 4.3 - AGENT
# =============================================================================

def run_smolagents_same_question() -> str:
    agent = CodeAgent(
        tools=[check_fridge, get_recipe, check_dietary_info],
        model=model,
        max_steps=5,
    )

    question = (
        "Je veux un dîner pour 2 personnes, rapide (<= 25 min), sans arachides, "
        "et idéalement riche en protéines. Regarde d'abord ce qu'il y a dans le frigo, "
        "puis propose 2 options et donne une recette détaillée pour celle que tu recommandes."
    )

    result = agent.run(question)
    print(result)
    return str(result)


if __name__ == "__main__":
    run_smolagents_same_question()
