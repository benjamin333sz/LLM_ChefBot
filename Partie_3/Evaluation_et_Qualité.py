from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv
from groq import Groq
from langfuse import Evaluation, get_client, observe

load_dotenv()
groq_client = Groq()

# =============================================================================
# 3.1 - CREATE DATASET
# =============================================================================

def create_chefbot_dataset() -> None:
    """
    Create Langfuse dataset: chefbot-menu-eval
    Each item:
      - input: {"constraints": "..."} (string that describes constraints)
      - expected_output: {"must_avoid": [...], "must_include": [...]} (+ optional criteria)
    """
    client = get_client()

    # Create dataset (if already exists, we just reuse it)
    try:
        client.create_dataset(
            name="chefbot-menu-eval",
            description="Evaluation dataset for ChefBot menu planning (constraints-based)",
            metadata={
                "created_by": "chefbot_eval_script",
                "domain": "meal_planning",
                "version": "1.0",
            },
        )
        print("✓ Dataset created: chefbot-menu-eval")
    except Exception as e:
        # Langfuse may throw if dataset exists (depending on version/config)
        print(f"ℹ Dataset may already exist. Continuing. ({e})")

    test_cases = [
        {
            "input": {"constraints": "Repas pour diabétique, dîner léger. Éviter sucre, soda, pâtes blanches. Inclure légumes verts et protéines maigres. Max 600 kcal/repas."},
            "expected_output": {
                "must_avoid": ["sucre", "soda", "pâtes blanches"],
                "must_include": ["légumes verts", "protéines maigres"],
                "max_calories_per_meal": 600,
            },
            "metadata": {"category": "health_diabetic_light"},
        },
        {
            "input": {"constraints": "Allergie sévère aux arachides + régime vegan. Budget 5€ par personne. Inclure légumineuses. Éviter arachide, beurre de cacahuète, miel, œuf, lait."},
            "expected_output": {
                "must_avoid": ["arachide", "cacahuète", "beurre de cacahuète", "miel", "œuf", "lait"],
                "must_include": ["légumineuses"],
                "budget_per_person_eur_max": 5,
            },
            "metadata": {"category": "allergy_peanut_vegan_budget"},
        },
        {
            "input": {"constraints": "Menu pour 6 convives style méditerranéen. Inclure huile d'olive, tomates, herbes (basilic/origan). Éviter porc. Option sans gluten si possible (éviter blé, farine de blé)."},
            "expected_output": {
                "must_avoid": ["porc", "blé", "farine de blé"],
                "must_include": ["huile d'olive", "tomates", "basilic", "origan"],
                "servings": 6,
            },
            "metadata": {"category": "cultural_mediterranean_group"},
        },
        {
            "input": {"constraints": "Régime pauvre en sel (hypertension). Éviter sel, sauce soja, charcuterie. Inclure épices/aromates (citron, ail) et légumes. Recette simple en 20 minutes."},
            "expected_output": {
                "must_avoid": ["sel", "sauce soja", "charcuterie"],
                "must_include": ["citron", "ail", "légumes"],
                "max_minutes": 20,
            },
            "metadata": {"category": "low_sodium_quick"},
        },
        {
            "input": {"constraints": "Préférences: cuisine japonaise maison. Inclure riz, gingembre. Éviter poisson cru (grossesse). Idée bento. Éviter alcool (mirin/saké)."},
            "expected_output": {
                "must_avoid": ["poisson cru", "mirin", "saké", "alcool"],
                "must_include": ["riz", "gingembre"],
                "style": "japonais",
            },
            "metadata": {"category": "cultural_japanese_pregnancy"},
        },
    ]

    # Add dataset items
    for case in test_cases:
        client.create_dataset_item(
            dataset_name="chefbot-menu-eval",
            input=case["input"],
            expected_output=case["expected_output"],
            metadata=case["metadata"],
        )

    print(f"✓ Added {len(test_cases)} items to chefbot-menu-eval")


# =============================================================================
# THE TASK (your planner)
# =============================================================================

@observe()
def chefbot_planner(constraints: str) -> str:
    """
    The function under test: returns a menu / recipe text (string).
    Replace the prompt/model as needed to match your ChefBot.
    """
    system = (
        "Tu es ChefBot. Propose un menu/recette en français qui respecte STRICTEMENT les contraintes.\n"
        "Réponds en texte clair, avec: titre(s), ingrédients, étapes courtes. "
        "Évite de mentionner des ingrédients interdits."
    )

    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": constraints},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


# =============================================================================
# 3.2 - PROGRAMMATIC EVALUATOR
# =============================================================================

def _normalize(text: str) -> str:
    # simple normalization for matching
    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def rule_evaluator(output: str, expected: dict) -> dict:
    """
    Checks automatically:
      - must_avoid not mentioned -> 0 or 1
      - must_include mentioned -> proportional score
    Returns a dict of scores.
    """
    out = _normalize(output or "")
    must_avoid = [str(x).lower() for x in expected.get("must_avoid", [])]
    must_include = [str(x).lower() for x in expected.get("must_include", [])]

    # must_avoid: fail if ANY forbidden ingredient appears
    forbidden_hits = [x for x in must_avoid if x and x in out]
    avoid_score = 1.0 if len(forbidden_hits) == 0 else 0.0

    # must_include: proportional
    include_hits = [x for x in must_include if x and x in out]
    include_score = 1.0 if len(must_include) == 0 else (len(include_hits) / len(must_include))

    # Optional: overall
    overall = (avoid_score + include_score) / 2.0

    return {
        "must_avoid_ok": avoid_score,
        "must_include_coverage": include_score,
        "overall_rules": overall,
        "debug_forbidden_hits": forbidden_hits,
        "debug_include_hits": include_hits,
    }


# =============================================================================
# 3.3 - LLM JUDGE
# =============================================================================

JUDGE_PROMPT = """Tu es un juge impartial qui évalue une réponse de ChefBot.

Tu reçois:
- question: les contraintes utilisateur
- output: la réponse produite
- expected: les critères attendus (must_avoid/must_include + autres champs potentiels)

Note chaque critère entre 0.0 et 1.0:
1) pertinence: respect des contraintes (dont must_avoid/must_include)
2) creativite: variété/originalité des recettes (sans trahir les contraintes)
3) praticite: faisable par un non-professionnel (ingrédients accessibles, étapes réalistes)

Réponds UNIQUEMENT en JSON strict:
{
  "pertinence": 0.0,
  "creativite": 0.0,
  "praticite": 0.0,
  "explanation": "une phrase courte"
}
Aucun texte hors JSON.
"""

@observe(name="llm-judge", as_type="generation")
def llm_judge(question: str, output: str, expected: dict) -> dict:
    user_message = (
        f"question:\n{question}\n\n"
        f"output:\n{output}\n\n"
        f"expected:\n{json.dumps(expected, ensure_ascii=False)}"
    )

    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )

    content = resp.choices[0].message.content.strip()

    # Best-effort strict JSON parse:
    # If the model wraps JSON in fences, we extract the first {...}
    if not content.startswith("{"):
        m = re.search(r"\{.*\}", content, flags=re.S)
        if m:
            content = m.group(0)

    return json.loads(content)


# =============================================================================
# 3.4 - RUN EXPERIMENT (Langfuse)
# =============================================================================

def run_experiment() -> Any:
    client = get_client()
    dataset = client.get_dataset("chefbot-menu-eval")

    # Task wrapper for run_experiment API
    def task(*, item) -> str:
        return chefbot_planner(item.input["constraints"])

    # Evaluator 1: rule-based -> list[Evaluation]
    def rules_eval(**kwargs) -> List[Evaluation]:
        output = kwargs.get("output")              # string
        expected_output = kwargs.get("expected_output")  # dict
        scores = rule_evaluator(output=output, expected=expected_output)

        # Don't log debug fields as scores; keep only numeric ones.
        return [
            Evaluation(name="must_avoid_ok", value=float(scores["must_avoid_ok"]),
                       comment=f"forbidden_hits={scores['debug_forbidden_hits']}"),
            Evaluation(name="must_include_coverage", value=float(scores["must_include_coverage"]),
                       comment=f"include_hits={scores['debug_include_hits']}"),
            Evaluation(name="overall_rules", value=float(scores["overall_rules"])),
        ]

    # Evaluator 2: LLM judge -> list[Evaluation]
    def llm_eval(**kwargs) -> List[Evaluation]:
        output = kwargs.get("output")
        expected_output = kwargs.get("expected_output")
        input_data = kwargs.get("input")

        judge = llm_judge(
            question=input_data["constraints"],
            output=output,
            expected=expected_output,
        )

        return [
            Evaluation(name="pertinence", value=float(judge["pertinence"]), comment=judge.get("explanation")),
            Evaluation(name="creativite", value=float(judge["creativite"])),
            Evaluation(name="praticite", value=float(judge["praticite"])),
        ]

    exp_name = f"chefbot-menu-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    results = client.run_experiment(
        name=exp_name,
        data=dataset.items,
        task=task,
        evaluators=[rules_eval, llm_eval],
        description="ChefBot menu planning evaluated by rules + LLM judge",
        metadata={
            "planner_model": "openai/gpt-oss-120b",
            "judge_model": "openai/gpt-oss-120b",
            "temperature": 0.4,
        },
    )

    print("\n✓ Experiment complete! Check Langfuse UI:")
    print("  Datasets > chefbot-menu-eval > Runs (ou Experiments selon ton UI)")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("CHEFBOT - DATASET + EVALUATION + EXPERIMENT")
    print("=" * 60)

    # 1) Create dataset (run once)
    create_chefbot_dataset()

    # 2) Run experiment
    run_experiment()

    # Flush traces/scores
    get_client().flush()
    print("✓ Flushed to Langfuse")
