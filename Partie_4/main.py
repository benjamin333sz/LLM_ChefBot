from __future__ import annotations

import json
from typing import Dict, Any, List

from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client
from smolagents import tool, ToolCallingAgent
from llm_utils import get_groq_litellm_model

load_dotenv()
groq_client = Groq()

# =============================================================================
# 4.1 - DEFINE 3 TOOLS (schemas for the LLM)
# =============================================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_fridge",
            "description": "Return the list of ingredients currently available in the fridge (simulated).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recipe",
            "description": "Return a detailed recipe for a given dish name (simulated).",
            "parameters": {
                "type": "object",
                "properties": {
                    "dish_name": {
                        "type": "string",
                        "description": "Dish name, e.g. 'Shakshuka' or 'Pasta Primavera'",
                    }
                },
                "required": ["dish_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_dietary_info",
            "description": "Return nutritional + allergen info for a given ingredient (simulated).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredient": {
                        "type": "string",
                        "description": "Ingredient name, e.g. 'peanut', 'egg', 'milk', 'tofu'",
                    }
                },
                "required": ["ingredient"],
            },
        },
    },
]

# =============================================================================
# 4.1 - TOOL IMPLEMENTATIONS (simulated data)
# =============================================================================

@observe()
def check_fridge() -> str:
    data = {
        "available": [
            "tomates",
            "oignons",
            "ail",
            "riz",
            "lentilles",
            "pois chiches",
            "tofu",
            "œufs",
            "yaourt",
            "citron",
            "huile d'olive",
            "basilic",
            "origan",
            "poivrons",
            "épinards",
        ]
    }
    return json.dumps(data, ensure_ascii=False)


@observe()
def get_recipe(dish_name: str) -> str:
    recipes = {
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
    }

    key = dish_name.strip().lower()
    recipe = recipes.get(key)

    if not recipe:
        recipe = {
            "title": dish_name.strip(),
            "ingredients": ["(ingrédients non disponibles dans la base simulée)"],
            "steps": ["(recette non disponible — propose un plat différent)"],
            "time_minutes": None,
        }

    return json.dumps(recipe, ensure_ascii=False)


@observe()
def check_dietary_info(ingredient: str) -> str:
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


TOOL_REGISTRY = {
    "check_fridge": check_fridge,
    "get_recipe": get_recipe,
    "check_dietary_info": check_dietary_info,
}

# =============================================================================
# 4.2 - MANUAL TOOL-CALLING LOOP (max 5 iterations)
# =============================================================================

@observe()
def manual_tool_calling_agent(user_message: str) -> str:
    """
    - send user msg + tool schemas
    - execute requested tool calls
    - return final answer when no tool_calls
    - max 5 iterations
    """

    messages = [
        {
            "role": "system",
            "content": (
                "Tu es ChefBot. Tu as accès à des outils via le mécanisme officiel de tool calling.\n"
                "RÈGLE ABSOLUE :\n"
                "- N'écris JAMAIS de balises ou texte du style <function=...>.\n"
                "- Si tu veux appeler un outil, tu dois UNIQUEMENT utiliser un tool_call structuré.\n"
                "- Après avoir reçu les résultats des outils, tu donnes une réponse finale claire.\n"
                "Objectif : respecter la question, utiliser les outils quand nécessaire."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    for iteration in range(5):
        print(f"\n[Iteration {iteration + 1}]")

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
                temperature=0.2,
            )
        except Exception as e:
            get_client().update_current_span(level="ERROR", status_message=str(e))
            raise

        msg = response.choices[0].message

        # ✅ Final answer
        if not getattr(msg, "tool_calls", None):
            return (msg.content or "").strip()

        # Add assistant tool-call message
        messages.append(msg)

        # Execute each tool call and append tool results
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args_raw = tool_call.function.arguments or "{}"

            try:
                args = json.loads(args_raw)
            except Exception:
                args = {}
                get_client().update_current_span(
                    level="ERROR",
                    status_message="Invalid JSON arguments from model",
                    metadata={"tool": name, "args_raw": args_raw[:500]},
                )

            print(f"Tool call: {name}({args})")

            func = TOOL_REGISTRY.get(name)
            if func:
                result = func(**args) if args else func()
            else:
                result = json.dumps({"error": f"unknown tool '{name}'"}, ensure_ascii=False)

            print(f"Result: {result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    return "Error: max iterations reached"


# =============================================================================
# 4.3 - TOOLS WITH @tool
# =============================================================================

@tool
def check_fridge() -> str:
    """
    Return the list of ingredients currently available in the fridge (simulated).

    Returns:
        JSON string: {"available": [...]}.
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
        dish_name: Name of the dish (e.g., "curry de pois chiches", "shakshuka").

    Returns:
        JSON string with: title, ingredients, steps, time_minutes.
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
        ingredient: Ingredient name (e.g., "arachide", "œufs", "tofu").

    Returns:
        JSON string with keys: ingredient, allergens, vegan, notes.
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


def run_smolagents_same_question() -> str:
    model = get_groq_litellm_model(
        model_id="groq/llama-3.3-70b-versatile",
        temperature=0.2,
    )

    agent = ToolCallingAgent(
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
    question = (
        "Je veux un dîner pour 2 personnes, rapide (<= 25 min), sans arachides, "
        "et idéalement riche en protéines. Regarde d'abord ce qu'il y a dans le frigo, "
        "puis propose 2 options et donne une recette détaillée pour celle que tu recommandes.\n"
        "Important : tu dois appeler get_recipe pour la recette finale que tu recommandes."
    )

    answer = manual_tool_calling_agent(question)
    print("\n=== FINAL ANSWER ===")
    print(answer)

    get_client().flush()
    run_smolagents_same_question()

