import json
import os
import sys


# Permet d'importer Partie_1 quand on lance Partie_2/main.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from Partie_1.chefbot import ask_chef  # Partie 1 (inchangée)
from Partie_2.planner import plan_weekly_menu


def run_part1_demo():
    prompt = "Que proposez-vous comme repas pour ce midi ?"
    temperatures = [0.1, 0.7, 1.2]
    for t in temperatures:
        print(f"\nTemperature = {t}\n" + "_" * 50)
        print(ask_chef(question=prompt, temperature=t))


def run_part2_demo():
    constraints = "Menu pour 2 personnes, budget moyen, cuisine de saison, sans porc, 2 repas végétariens, rapide en semaine."
    menu = plan_weekly_menu(constraints)
    print(json.dumps(menu, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Choisis ce que tu veux tester
    # run_part1_demo()
    run_part2_demo()
