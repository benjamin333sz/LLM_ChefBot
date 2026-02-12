import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client

load_dotenv()

groq_client = Groq()
langfuse = get_client()

MODEL_ID = "openai/gpt-oss-120b"


def _chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    """Appel LLM minimal et robuste : renvoie toujours un texte."""
    resp = groq_client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def _safe_json_loads(raw: str) -> Any:
    """
    JSON robuste : essaye d’extraire un bloc JSON même si le modèle ajoute du texte.
    """
    raw = raw.strip()

    # Cas simple : c'est du JSON pur
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Sinon, on tente d'extraire le premier objet/array JSON
    first_obj = raw.find("{")
    last_obj = raw.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        return json.loads(raw[first_obj:last_obj + 1])

    first_arr = raw.find("[")
    last_arr = raw.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        return json.loads(raw[first_arr:last_arr + 1])

    raise ValueError("JSON introuvable dans la réponse du modèle.")


@observe()
def ask_chef(question: str, temperature: float = 0.2) -> str:
    """
    Partie 1 : appel simple.
    """
    # Trace naming / tags (adaptables)
    try:
        langfuse.update_current_trace(
            name="Groupe XXX - Partie 1 - ask_chef",
            tags=["Groupe XXX", "Partie 1", "ChefBot"],
            metadata={"type": "single_call", "season": "auto"},
        )
    except Exception:
        # Si la version du SDK ne supporte pas ces champs, on ignore sans casser le TP.
        pass

    answer = _chat(
        messages=[
            {
                "role": "system",
                "content": "Tu es ChefBot, un chef cuisinier français spécialisé en cuisine de saison. Réponds de façon claire et pratique.",
            },
            {"role": "user", "content": question},
        ],
        temperature=temperature,
    )

    try:
        langfuse.update_current_span(metadata={"temperature": temperature})
    except Exception:
        pass

    langfuse.flush()
    return answer


@observe()
def _plan_steps(constraints: str) -> Dict[str, Any]:
    """
    Partie 2.1 — Étape 1 : Planification (doit retourner du JSON).
    JSON attendu :
    {
      "steps": [
        {"id": 1, "title": "...", "prompt": "..."},
        ...
      ]
    }
    """
    system = (
        "Tu es un planificateur culinaire. "
        "Tu DOIS répondre uniquement en JSON valide, sans texte autour."
    )
    user = f"""
Contraintes utilisateur:
{constraints}

Retourne un plan JSON avec une clé "steps" (liste d'étapes).
Chaque étape: id (int), title (str), prompt (str).
"""

    raw = _chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )

    try:
        plan = _safe_json_loads(raw)
        if not isinstance(plan, dict) or "steps" not in plan or not isinstance(plan["steps"], list):
            raise ValueError("Structure JSON invalide (clé steps manquante ou incorrecte).")

        # Metadata span
        try:
            langfuse.update_current_span(metadata={"num_steps": len(plan["steps"])})
        except Exception:
            pass

        return plan

    except Exception as e:
        # Log d'erreur Langfuse (niveau ERROR)
        try:
            langfuse.update_current_span(level="ERROR", status_message=str(e), metadata={"raw": raw[:800]})
        except Exception:
            pass
        raise


@observe()
def _plan_steps_with_retry(constraints: str) -> Dict[str, Any]:
    """
    Partie 2.2 — Gestion d’erreur : 1 retry max si JSON invalide.
    """
    try:
        return _plan_steps(constraints)
    except Exception:
        # Retry unique avec consigne plus stricte
        system = (
            "Tu as échoué à produire du JSON. "
            "Tu DOIS répondre UNIQUEMENT par un JSON valide, sans backticks, sans texte."
        )
        user = f"""
Contraintes utilisateur:
{constraints}

Retourne EXACTEMENT un JSON de la forme:
{{
  "steps": [
    {{"id": 1, "title": "...", "prompt": "..."}}
  ]
}}
"""

        raw = _chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
        )

        try:
            plan = _safe_json_loads(raw)
            if not isinstance(plan, dict) or "steps" not in plan or not isinstance(plan["steps"], list):
                raise ValueError("Structure JSON invalide même après retry.")

            try:
                langfuse.update_current_span(metadata={"retry_used": True, "num_steps": len(plan["steps"])})
            except Exception:
                pass

            return plan

        except Exception as e:
            try:
                langfuse.update_current_span(level="ERROR", status_message=str(e), metadata={"raw_retry": raw[:800]})
            except Exception:
                pass
            raise


@observe()
def _execute_step(step: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Partie 2.1 — Étape 2 : Exécution (1 appel LLM par étape).
    """
    step_title = str(step.get("title", "step"))
    step_prompt = str(step.get("prompt", ""))

    # On passe le contexte précédent au modèle
    system = (
        "Tu es ChefBot en mode exécution. "
        "Tu appliques l'étape demandée et tu donnes une sortie concise, réutilisable par les étapes suivantes."
    )
    user = f"""
Étape: {step_title}

Contexte déjà produit (JSON):
{json.dumps(context, ensure_ascii=False)}

Consigne:
{step_prompt}
"""

    out = _chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
    )

    try:
        langfuse.update_current_span(metadata={"step_id": step.get("id"), "step_title": step_title})
    except Exception:
        pass

    return out


@observe()
def _synthesize_menu(constraints: str, plan: Dict[str, Any], step_outputs: Dict[str, str]) -> Dict[str, Any]:
    """
    Partie 2.1 — Étape 3 : Synthèse (menu semaine cohérent) => dict (JSON).
    """
    system = (
        "Tu es ChefBot, chef cuisinier français spécialiste cuisine de saison. "
        "Tu dois produire UNIQUEMENT un JSON valide."
    )

    user = f"""
Contraintes:
{constraints}

Plan:
{json.dumps(plan, ensure_ascii=False)}

Résultats des étapes:
{json.dumps(step_outputs, ensure_ascii=False)}

Retourne un menu JSON pour 7 jours (lundi->dimanche) avec:
- day
- lunch
- dinner
- notes (optionnel)
Format attendu:
{{
  "weekly_menu": [
    {{"day":"Lundi","lunch":"...","dinner":"...","notes":"..."}}
  ]
}}
"""

    raw = _chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )

    try:
        menu = _safe_json_loads(raw)
        if not isinstance(menu, dict) or "weekly_menu" not in menu:
            raise ValueError("JSON final invalide (clé weekly_menu manquante).")

        return menu

    except Exception as e:
        try:
            langfuse.update_current_span(level="ERROR", status_message=str(e), metadata={"raw": raw[:800]})
        except Exception:
            pass
        raise


@observe()
def plan_weekly_menu(constraints: str) -> Dict[str, Any]:
    """
    Partie 2 — pipeline complet : Planification -> Exécution multi-étapes -> Synthèse.
    """
    # Trace naming / tags (adaptables)
    try:
        langfuse.update_current_trace(
            name="Groupe XXX - Partie 2 - plan_weekly_menu",
            tags=["Groupe XXX", "Partie 2", "ChefBot"],
            metadata={"type": "multi_step_menu"},
        )
    except Exception:
        pass

    plan = _plan_steps_with_retry(constraints)

    context: Dict[str, Any] = {"constraints": constraints}
    step_outputs: Dict[str, str] = {}

    for step in plan["steps"]:
        key = f"{step.get('id', len(step_outputs)+1)}_{step.get('title', 'step')}"
        out = _execute_step(step, context=context)
        step_outputs[key] = out

        # On enrichit le contexte pour les étapes suivantes
        context["step_outputs"] = step_outputs

    menu = _synthesize_menu(constraints=constraints, plan=plan, step_outputs=step_outputs)

    langfuse.flush()
    return menu


if __name__ == "__main__":
    # Petit test manuel Partie 2
    constraints = "Menu pour 2 personnes, budget moyen, cuisine de saison, sans porc, 2 repas végétariens, rapide en semaine."
    menu = plan_weekly_menu(constraints)
    print(json.dumps(menu, indent=2, ensure_ascii=False))
