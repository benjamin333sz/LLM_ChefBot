import json
from typing import Any, Dict

from langfuse import observe, get_client, propagate_attributes

from .llm_utils import chat, safe_json_loads, langfuse

GROUP = "Groupe_SZUREK_KUSNIEREK_GOSSELIN"


@observe(name=f"{GROUP}_Partie_2",as_type="chain")
def plan_weekly_menu(constraints: str) -> Dict[str, Any]:
    # Trace + tags groupe
    with propagate_attributes(tags=["Partie_2", GROUP]):
        get_client().update_current_span(metadata={"partie": "2", "status": "start"})

        plan = _plan_steps_with_retry(constraints)

        context: Dict[str, Any] = {"constraints": constraints}
        step_outputs: Dict[str, str] = {}

        # Exécution multi-étapes (1 call / step)
        for step in plan["steps"]:
            key = f"{step.get('id', len(step_outputs)+1)}_{step.get('title', 'step')}"
            out = _execute_step(step, context)
            step_outputs[key] = out
            context["step_outputs"] = step_outputs

        menu = _synthesize(constraints, plan, step_outputs)

        get_client().update_current_span(metadata={"status": "success"})
        langfuse.flush()
        return menu


@observe(name=f"plan")
def _plan_steps(constraints: str) -> Dict[str, Any]:
    with propagate_attributes(tags=["Partie_2", GROUP, "plan"]):
        system = "Tu es un planificateur. Réponds UNIQUEMENT en JSON valide."
        user = f"""
Contraintes:
{constraints}

Retourne un JSON EXACT:
{{
  "steps": [
    {{"id": 1, "title": "...", "prompt": "..."}}
  ]
}}
"""

        raw = chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )

        try:
            plan = safe_json_loads(raw)
            if not isinstance(plan, dict) or "steps" not in plan or not isinstance(plan["steps"], list):
                raise ValueError("Structure JSON invalide: clé 'steps' attendue.")

            get_client().update_current_span(metadata={"num_steps": len(plan["steps"]), "status": "success"})
            return plan

        except Exception as e:
            get_client().update_current_span(level="ERROR", status_message=str(e), metadata={"raw": raw[:800]})
            raise


@observe(name=f"plan_retry")
def _plan_steps_with_retry(constraints: str) -> Dict[str, Any]:
    with propagate_attributes(tags=["Partie_2", GROUP, "plan", "retry"]):
        try:
            return _plan_steps(constraints)
        except Exception:
            # 1 retry max, consigne plus stricte
            system = "Tu dois produire UNIQUEMENT un JSON valide. Aucun texte. Aucun backtick."
            user = f"""
Contraintes:
{constraints}

Retourne EXACTEMENT:
{{
  "steps": [
    {{"id": 1, "title": "...", "prompt": "..."}}
  ]
}}
"""
            raw = chat(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
            )

            try:
                plan = safe_json_loads(raw)
                if not isinstance(plan, dict) or "steps" not in plan or not isinstance(plan["steps"], list):
                    raise ValueError("JSON toujours invalide après retry.")

                get_client().update_current_span(metadata={"retry_used": True, "num_steps": len(plan["steps"])})
                return plan

            except Exception as e:
                get_client().update_current_span(level="ERROR", status_message=str(e), metadata={"raw_retry": raw[:800]})
                raise


@observe(name=f"execute_step")
def _execute_step(step: Dict[str, Any], context: Dict[str, Any]) -> str:
    with propagate_attributes(tags=["Partie_2", GROUP, "execute"]):
        step_title = str(step.get("title", "step"))
        step_prompt = str(step.get("prompt", ""))

        system = (
            "Tu es ChefBot en mode exécution. "
            "Tu appliques l'étape et tu renvoies une sortie concise réutilisable."
        )
        user = f"""
Étape: {step_title}

Contexte (JSON):
{json.dumps(context, ensure_ascii=False)}

Consigne:
{step_prompt}
"""

        out = chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
        )

        get_client().update_current_span(metadata={"step_id": step.get("id"), "step_title": step_title})
        return out


@observe(name=f"synthesis")
def _synthesize(constraints: str, plan: Dict[str, Any], step_outputs: Dict[str, str]) -> Dict[str, Any]:
    with propagate_attributes(tags=["Partie_2", GROUP, "synthesis"]):
        system = (
            "Tu es ChefBot, chef cuisinier français de saison. "
            "Réponds UNIQUEMENT en JSON valide."
        )
        user = f"""
Contraintes:
{constraints}

Plan:
{json.dumps(plan, ensure_ascii=False)}

Résultats:
{json.dumps(step_outputs, ensure_ascii=False)}

Retourne:
{{
  "weekly_menu": [
    {{"day":"Lundi","lunch":"...","dinner":"...","notes":"..."}}
  ]
}}
"""

        raw = chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )

        try:
            menu = safe_json_loads(raw)
            if not isinstance(menu, dict) or "weekly_menu" not in menu:
                raise ValueError("JSON final invalide: clé 'weekly_menu' manquante.")
            return menu
        except Exception as e:
            get_client().update_current_span(level="ERROR", status_message=str(e), metadata={"raw": raw[:800]})
            raise
