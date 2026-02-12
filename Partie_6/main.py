from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

from dotenv import load_dotenv
from smolagents import Tool, CodeAgent, LiteLLMModel, tool

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================

import litellm
litellm._turn_on_debug()

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY manquant. Mets-le dans .env puis relance.")

# Important: force provider + api_base => évite l’erreur "LLM Provider NOT provided"
MODEL_ID = "llama-3.3-70b-versatile"
model = LiteLLMModel(
    model_id=MODEL_ID,
    custom_llm_provider="groq",
    api_key=os.getenv("GROQ_API_KEY"),
    api_base="https://api.groq.com/openai/v1",
)

TRACE_FILE = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
def trace(line: str) -> None:
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


# =============================================================================
# TOOLS (réutilisés / simulés) — Partie 4
# =============================================================================

@tool
def check_fridge() -> str:
    """
    Return the list of ingredients currently available in the fridge (simulated).

    Returns:
        JSON string: {"available": [...]}.
    """
    return json.dumps(
        {
            "available": [
                "tomates", "oignons", "ail", "citron", "huile d'olive",
                "basilic", "poivrons", "épinards",
                "lentilles", "pois chiches", "riz",
                "yaourt",
                "fruits (pommes, poires, oranges)",
            ]
        },
        ensure_ascii=False,
    )


@tool
def get_recipe(dish_name: str) -> str:
    """
    Return a detailed recipe for a given dish name (simulated).

    Args:
        dish_name: Name of the dish (e.g., "houmous").

    Returns:
        JSON string describing a recipe: {title, ingredients, steps, time_minutes}.
    """
    recipes = {
        "houmous": {
            "title": "Houmous citronné",
            "ingredients": ["pois chiches", "citron", "huile d'olive", "ail", "cumin", "sel"],
            "steps": ["Mixer tous les ingrédients.", "Ajuster sel/citron.", "Servir avec légumes crus."],
            "time_minutes": 10,
        },
        "curry de pois chiches": {
            "title": "Curry de pois chiches & épinards",
            "ingredients": ["pois chiches", "tomates", "oignons", "ail", "épinards", "épices", "huile d'olive"],
            "steps": [
                "Faire revenir oignons + ail + épices.",
                "Ajouter tomates + pois chiches, mijoter 12 min.",
                "Ajouter épinards 2 min. Servir.",
            ],
            "time_minutes": 20,
        },
        "salade de fruits": {
            "title": "Salade de fruits frais",
            "ingredients": ["fruits", "citron"],
            "steps": ["Couper les fruits.", "Ajouter un trait de citron.", "Servir frais."],
            "time_minutes": 10,
        },
    }
    key = dish_name.strip().lower()
    return json.dumps(
        recipes.get(
            key,
            {
                "title": dish_name.strip(),
                "ingredients": [],
                "steps": ["Recette non disponible (simulé)."],
                "time_minutes": None,
            },
        ),
        ensure_ascii=False,
    )


@tool
def check_dietary_info(ingredient: str) -> str:
    """
    Return allergen/nutrition flags for an ingredient (simulated).

    Args:
        ingredient: Ingredient name to check.

    Returns:
        JSON string: {ingredient, allergens, vegan, notes}.
    """
    db = {
        "arachide": {"allergens": ["arachide", "fruits_a_coque"], "vegan": True, "notes": "Allergène majeur."},
        "noix": {"allergens": ["fruits_a_coque"], "vegan": True, "notes": "Allergène majeur."},
        "amande": {"allergens": ["fruits_a_coque"], "vegan": True, "notes": "Allergène majeur."},
        "noisette": {"allergens": ["fruits_a_coque"], "vegan": True, "notes": "Allergène majeur."},
        "blé": {"allergens": ["gluten"], "vegan": True, "notes": "Contient gluten."},
        "gluten": {"allergens": ["gluten"], "vegan": True, "notes": "Allergène."},
        "riz": {"allergens": [], "vegan": True, "notes": "Naturellement sans gluten."},
        "pois chiches": {"allergens": [], "vegan": True, "notes": "Protéines végétales."},
        "lentilles": {"allergens": [], "vegan": True, "notes": "Protéines + fibres."},
        "yaourt": {"allergens": ["lait"], "vegan": False, "notes": "Produit laitier."},
    }
    key = ingredient.strip().lower()
    info = db.get(key, {"allergens": [], "vegan": None, "notes": "Info non disponible (simulé)."})
    return json.dumps({"ingredient": ingredient, **info}, ensure_ascii=False)


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a basic math expression and return the result.

    Args:
        expression: Expression using only numbers and + - * / . ( ) spaces.

    Returns:
        Result as string or "Invalid expression".
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Invalid expression"
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid expression"


# =============================================================================
# TOOL (réutilisé) — Partie 5 : MenuDatabaseTool (class Tool)
# =============================================================================

@dataclass
class Dish:
    name: str
    price: float
    prep_minutes: int
    allergens: List[str]
    category: str
    tags: List[str]


class MenuDatabaseTool(Tool):
    name = "menu_database"
    description = "Search the restaurant menu by criteria and return JSON results."
    inputs = {
        "category": {"type": "string", "description": "Optional: entrée/plat/dessert/boisson or 'all'", "nullable": True},
        "max_price": {"type": "number", "description": "Optional: max price per dish", "nullable": True},
        "exclude_allergens": {"type": "array", "description": "Optional: allergens to exclude", "items": {"type": "string"}, "nullable": True},
        "include_tags": {"type": "array", "description": "Optional: required tags", "items": {"type": "string"}, "nullable": True},
        "limit": {"type": "number", "description": "Optional: max results", "nullable": True},
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        # Menu réduit safe: seulement vegan/sans gluten
        self.dishes = [
            Dish("Houmous & légumes", 5.5, 10, [], "entrée", ["vegan", "sans_gluten"]),
            Dish("Curry pois chiches", 16.0, 20, [], "plat", ["vegan", "sans_gluten"]),
            Dish("Salade fruits", 6.5, 8, [], "dessert", ["vegan", "sans_gluten"]),
            Dish("Eau petillante", 2.5, 1, [], "boisson", ["vegan", "sans_gluten"]),
        ]

    def forward(
        self,
        category: Optional[str] = None,
        max_price: Optional[float] = None,
        exclude_allergens: Optional[List[str]] = None,
        include_tags: Optional[List[str]] = None,
        limit: Optional[float] = 10,
    ) -> str:
        cat = (category or "").strip().lower()
        if cat in ("", "all", "*"):
            cat = ""

        exc = [(x or "").strip().lower() for x in (exclude_allergens or []) if x]
        inc = [(x or "").strip().lower().replace(" ", "_") for x in (include_tags or []) if x]

        results = []
        for d in self.dishes:
            if cat and d.category.lower() != cat:
                continue
            if max_price is not None and d.price > float(max_price):
                continue
            if exc and any(a in [aa.lower() for aa in d.allergens] for a in exc):
                continue
            if inc and any(t not in [tt.lower() for tt in d.tags] for t in inc):
                continue
            results.append(d)

        results.sort(key=lambda x: x.price)
        if limit is not None:
            results = results[: int(limit)]

        payload = [
            {"name": r.name, "price": r.price, "prep_minutes": r.prep_minutes, "allergens": r.allergens, "category": r.category, "tags": r.tags}
            for r in results
        ]
        return json.dumps({"results": payload}, ensure_ascii=False)


# =============================================================================
# 6.1 - MULTI AGENT SYSTEM
# =============================================================================

def build_agents():
    menu_tool = MenuDatabaseTool()

    nutritionist = CodeAgent(
        tools=[check_dietary_info],
        model=model,
        max_steps=3,
        instructions=(
            "Nutritionist.\n"
            "Vérifie: pas gluten, pas fruits_a_coque, pas arachide.\n"
            "OK ou fixes nécessaires?"
        ),
    )

    chef_agent = CodeAgent(
        tools=[check_fridge, get_recipe],
        model=model,
        max_steps=3,
        instructions=(
            "Chef.\n"
            "2 idées/service: apéro, entrée, plat, dessert.\n"
            "Tout le monde mange chaque service (vegan/sans gluten).\n"
            "JSON: {aperitif:[...], entree:[...], plat:[...], dessert:[...]}."
        ),
    )

    budget_agent = CodeAgent(
        tools=[calculate, menu_tool],
        model=model,
        max_steps=3,
        instructions=(
            "Budget.\n"
            "Menu 8 pers, 120€ max.\n"
            "Exclure: gluten, fruits_a_coque, arachide.\n"
            "Tag: vegan + sans_gluten si possible.\n"
            "JSON: {menu:{...}, total_eur:...}."
        ),
    )

    manager = CodeAgent(
        tools=[],
        model=model,
        max_steps=2,
        instructions=(
            "Manager synthétise final.\n"
            "Menu + vérif contraintes + budget."
        ),
    )

    return manager, nutritionist, chef_agent, budget_agent


def manager_run(user_request: str) -> str:
    manager, nutritionist, chef_agent, budget_agent = build_agents()

    trace("=" * 80)
    trace("PARTIE 6 - MULTI AGENT RUN")
    trace(user_request)
    trace("=" * 80)

    # 1) Chef
    chef_prompt = (
        f"{user_request}\n\n"
        "Donne 2 idées par service (apéro, entrée, plat, dessert) compatibles avec toutes les contraintes.\n"
        "Réponds en JSON strict: {aperitif:[...], entree:[...], plat:[...], dessert:[...]}."
    )
    chef_out = chef_agent.run(chef_prompt)
    trace("\n[chef_agent]\n" + str(chef_out))

    # 2) Budget
    budget_prompt = (
        f"{user_request}\n\n"
        "Construis un menu complet via menu_database.\n"
        "Hypothèse simple: on prend 8 portions par service (une par personne).\n"
        "Exclus gluten + fruits_a_coque + arachide.\n"
        "Utilise include_tags=['vegan','sans_gluten'] si nécessaire.\n"
        "Calcule total = somme(prix_plat * 8) pour chaque service.\n"
        "Réponds en JSON strict: {menu:{aperitif:..., entree:..., plat:..., dessert:...}, breakdown:{...}, total_eur:..., margin_eur:...}."
    )
    budget_out = budget_agent.run(budget_prompt)
    trace("\n[budget_agent]\n" + str(budget_out))

    # 3) Nutritionist
    nutrition_prompt = (
        f"{user_request}\n\n"
        f"Propositions chef: {chef_out}\n"
        f"Menu budget: {budget_out}\n\n"
        "Valide strictement: pas de gluten, pas de fruits à coque/arachide.\n"
        "Tout doit convenir aux végétariens (donc pas de viande/poisson).\n"
        "Réponds en JSON strict: {ok: true/false, issues:[...], fixes:[...], notes:[...]}."
    )
    nutri_out = nutritionist.run(nutrition_prompt)
    trace("\n[nutritionist]\n" + str(nutri_out))

    # 4) Manager final (no tools)
    manager_prompt = (
        "Synthétise une proposition finale au client.\n\n"
        f"DEMANDE:\n{user_request}\n\n"
        f"CHEF:\n{chef_out}\n\n"
        f"BUDGET:\n{budget_out}\n\n"
        f"NUTRITION:\n{nutri_out}\n\n"
        "Réponse finale en français, structurée:\n"
        "1) Menu final (apéro/entrée/plat/dessert)\n"
        "2) Compatibilité contraintes\n"
        "3) Budget (total + marge)\n"
        "4) Option(s) faciles si besoin\n"
    )
    final = manager.run(manager_prompt)
    trace("\n[manager_final]\n" + str(final))

    return str(final)


# =============================================================================
# 6.2 - TEST
# =============================================================================

if __name__ == "__main__":
    request = (
        "Je reçois 8 personnes samedi soir. Parmi eux : 2 vegetariens, 1 intolerant au gluten, "
        "1 allergique aux fruits a coque. Budget total : 120 euros. "
        "Je veux un aperitif, une entree, un plat principal et un dessert. "
        "Il faut que tout le monde puisse manger chaque service."
    )

    print("\n=== PARTIE 6 - MULTI AGENT ===\n")
    answer = manager_run(request)
    print(answer)
    print("\nTrace saved in:", TRACE_FILE)
