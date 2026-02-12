from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

from dotenv import load_dotenv
from smolagents import Tool, CodeAgent, LiteLLMModel, tool

load_dotenv()

# -----------------------------------------------------------------------------
# MODEL (adapte si besoin)
# -----------------------------------------------------------------------------
model = LiteLLMModel(model_id="groq/llama-3.3-70b-versatile")

# -----------------------------------------------------------------------------
# SIMPLE TXT TRACING
# -----------------------------------------------------------------------------
TRACE_FILE = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def trace(text: str) -> None:
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


# =============================================================================
# 5.1 - MENU DATABASE TOOL
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
    description = (
        "Search the restaurant menu by criteria: category, max price, exclude allergens, include tags. "
        "Returns matching dishes as JSON."
    )

    # Optional parameters => nullable=True
    inputs = {
        "category": {
            "type": "string",
            "description": "Optional. One of: entrée, plat, dessert, boisson. Use 'all' to mean no filter.",
            "nullable": True,
        },
        "max_price": {
            "type": "number",
            "description": "Optional. Maximum price per dish.",
            "nullable": True,
        },
        "exclude_allergens": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional. Allergens to exclude (e.g. gluten, lait, œuf, soja).",
            "nullable": True,
        },
        "include_tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional. Tags required. Use: vegetarien, vegan, sans_gluten.",
            "nullable": True,
        },
        "limit": {
            "type": "number",
            "description": "Optional. Max number of results.",
            "nullable": True,
        },
    }

    output_type = "string"

    # Canonical sets (helps robust filtering)
    _VALID_CATEGORIES = {"entrée", "plat", "dessert", "boisson"}
    _VALID_TAGS = {"vegetarien", "vegan", "sans_gluten"}
    _VALID_ALLERGENS = {"gluten", "lait", "œuf", "soja", "arachide"}

    # Synonyms (LLM often outputs english)
    _TAG_SYNONYMS = {
        "vegetarian": "vegetarien",
        "vegetarien": "vegetarien",
        "veggie": "vegetarien",
        "vegan": "vegan",
        "gluten-free": "sans_gluten",
        "gluten free": "sans_gluten",
        "sans gluten": "sans_gluten",
        "sans_gluten": "sans_gluten",
        "gf": "sans_gluten",
    }

    _ALLERGEN_SYNONYMS = {
        "egg": "œuf",
        "oeuf": "œuf",
        "œuf": "œuf",
        "milk": "lait",
        "dairy": "lait",
        "lait": "lait",
        "soy": "soja",
        "soya": "soja",
        "soja": "soja",
        "peanut": "arachide",
        "arachide": "arachide",
        "gluten": "gluten",
        "wheat": "gluten",
        "blé": "gluten",
    }

    def __init__(self):
        super().__init__()
        self.dishes = [
            Dish("Salade quinoa", 8.5, 12, [], "entrée", ["vegan", "sans_gluten"]),
            Dish("Curry pois chiches", 16.0, 20, [], "plat", ["vegan", "sans_gluten"]),
            Dish("Risotto champignons", 18.0, 25, ["lait"], "plat", ["vegetarien"]),
            Dish("Saumon grillé", 21.0, 22, [], "plat", ["sans_gluten"]),
            Dish("Salade fruits", 6.5, 8, [], "dessert", ["vegan", "sans_gluten"]),
            Dish("Thé vert", 3.0, 3, [], "boisson", ["vegan", "sans_gluten"]),
        ]
        trace(f"[MenuDatabaseTool] init dishes={len(self.dishes)}")

    def _norm_category(self, category: Optional[str]) -> Optional[str]:
        if not category:
            return None
        c = category.strip().lower()
        if c in ("all", "*", "toutes", "tout"):
            return None
        # accept accents or not
        c = c.replace("entree", "entrée")
        if c in self._VALID_CATEGORIES:
            return c
        # If unknown, ignore category filter (prevents empty results)
        return None

    def _norm_tags(self, tags: Optional[List[str]]) -> List[str]:
        if not tags:
            return []
        out = []
        for t in tags:
            key = t.strip().lower()
            mapped = self._TAG_SYNONYMS.get(key, key)
            mapped = mapped.replace(" ", "_")
            if mapped in self._VALID_TAGS:
                out.append(mapped)
        return out

    def _norm_allergens(self, allergens: Optional[List[str]]) -> List[str]:
        if not allergens:
            return []
        out = []
        for a in allergens:
            key = a.strip().lower()
            mapped = self._ALLERGEN_SYNONYMS.get(key, key)
            if mapped in self._VALID_ALLERGENS:
                out.append(mapped)
        return out

    def forward(
        self,
        category: Optional[str] = None,
        max_price: Optional[float] = None,
        exclude_allergens: Optional[List[str]] = None,
        include_tags: Optional[List[str]] = None,
        limit: Optional[float] = 10,
    ) -> str:
        cat = self._norm_category(category)
        exc_all = self._norm_allergens(exclude_allergens)
        inc_tags = self._norm_tags(include_tags)

        results: List[Dish] = []
        for d in self.dishes:
            if cat and d.category.lower() != cat:
                continue

            if max_price is not None and d.price > float(max_price):
                continue

            if exc_all:
                dish_all = [x.lower() for x in d.allergens]
                if any(a in dish_all for a in exc_all):
                    continue

            if inc_tags:
                dish_tags = [x.lower() for x in d.tags]
                if any(t not in dish_tags for t in inc_tags):
                    continue

            results.append(d)

        results.sort(key=lambda x: x.price)
        if limit is not None:
            results = results[: int(limit)]

        payload = [
            {
                "name": d.name,
                "price": d.price,
                "prep_minutes": d.prep_minutes,
                "allergens": d.allergens,
                "category": d.category,
                "tags": d.tags,
            }
            for d in results
        ]

        trace(
            f"[menu_database] category={category}->{cat} max_price={max_price} "
            f"exclude_allergens={exclude_allergens}->{exc_all} include_tags={include_tags}->{inc_tags} "
            f"results={len(payload)}"
        )

        return json.dumps({"results": payload}, ensure_ascii=False)


# =============================================================================
# TOOL: CALCULATE
# =============================================================================

@tool
def calculate(expression: str) -> str:
    """
    Evaluate a math expression.

    Args:
        expression: Math expression like '18 + 16 + 7'
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Invalid expression"
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid expression"


# =============================================================================
# 5.2 - AGENT WITH PLANNING
# =============================================================================

def build_agent() -> CodeAgent:
    # NOTE: tool instance created here to be shared in conversation as well
    menu_tool = MenuDatabaseTool()

    return CodeAgent(
        tools=[menu_tool, calculate],
        model=model,
        planning_interval=2,
        max_steps=5,
        instructions=(
            "Tu es serveur.\n"
            "Propose menu 3 pers: entrée + plat + dessert.\n"
            "- Végétarien (tag 'vegetarien')\n"
            "- Sans gluten (exclude 'gluten')\n"
            "- Sans contrainte\n"
            "Budget 60€. Utilise menu_database et calculate.\n"
        ),
    )


def test_planning_agent() -> None:
    agent = build_agent()

    question = (
        "On est 3. Un vegetarien, un sans gluten, et moi je mange de tout. "
        "Budget max 60 euros pour le groupe. Proposez-nous un menu complet."
    )

    trace("\n--- 5.2 TEST (planning agent) ---")
    trace("USER: " + question)

    result = agent.run(question)
    trace("AGENT: " + str(result))

    print(result)


# =============================================================================
# 5.3 - CONVERSATIONAL AGENT (reset=False)
# =============================================================================

def test_conversation() -> None:
    agent = build_agent()

    trace("\n--- 5.3 TEST (conversation, 3 turns) ---")

    # Turn 1
    q1 = "Bonsoir ! On est 3 (1 végétarien, 1 sans gluten, 1 sans contrainte). Tu nous suggères quoi ?"
    trace("USER(1): " + q1)
    r1 = agent.run(q1)
    trace("AGENT(1): " + str(r1))
    print("\nTour 1:\n", r1)

    # Turn 2
    q2 = "Finalement le végétarien ne veut pas de risotto. Tu remplaces son plat par autre chose."
    trace("USER(2): " + q2)
    r2 = agent.run(q2, reset=False)
    trace("AGENT(2): " + str(r2))
    print("\nTour 2:\n", r2)

    # Turn 3
    q3 = "Ok, maintenant fais l'addition détaillée finale pour les 3."
    trace("USER(3): " + q3)
    r3 = agent.run(q3, reset=False)
    trace("AGENT(3): " + str(r3))
    print("\nTour 3:\n", r3)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    print("\n=== PARTIE 5 ===\n")
    trace("=" * 70)
    trace("RESTAURANT INTELLIGENT - RUN TRACE")
    trace(f"timestamp={datetime.now().isoformat()}")
    trace("=" * 70)

    test_planning_agent()
    test_conversation()

    print("\nTrace saved in:", TRACE_FILE)
