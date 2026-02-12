# TP - ChefBot : De l'Appel API au Systeme Multi-Agent

> Vous allez construire **ChefBot**, un assistant culinaire intelligent qui evolue au fil du TP.
> On part d'un simple appel LLM pour arriver a un systeme multi-agent capable de planifier des repas,
> gerer un inventaire, et s'auto-evaluer. Chaque partie construit sur la precedente.

## Prerequis

Fichier `.env` requis :
```
GROQ_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY_RENDU="pk-lf-376e7285-599e-4979-a5dc-fe99f4b1229a"
LANGFUSE_SECRET_KEY_RENDU="sk-lf-8b6e288f-2592-47ee-8c21-2ca2479d92fa"
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Partie 1 - Premier Contact

*Concepts : appel LLM basique, system prompt, observabilite*

### 1.1 - Mon premier appel LLM

Creez un fichier `chefbot.py`. Ecrivez une fonction `ask_chef(question: str) -> str` qui :
- Utilise le client Groq
- A un system prompt definissant ChefBot comme un chef cuisinier francais specialise en cuisine de saison
- Prend une question utilisateur en parametre
- Retourne la reponse du modele


### 1.2 - Ajout de l'observabilite

Modifiez votre code pour integrer Langfuse :
- Decorez `ask_chef` avec `@observe()`
- Ajoutez des metadata sur la trace (ex : `type`, `season`)
- Appelez `get_client().flush()` a la fin

Verifiez dans l'interface Langfuse que votre trace apparait bien.

### 1.3 - Jouez avec la temperature

Appelez `ask_chef` **3 fois** avec la meme question mais des temperatures differentes (0.1, 0.7, 1.2).
Observez les differences dans les reponses. Notez vos observations en commentaire dans le code.

**Livrables** : 
- `ask_chef` fonctionne, les traces sont visibles dans Langfuse.
- Vos traces sont nomées et taggées avec votre nom de groupe (Groupe XXX, Partie 1)
- Une fois que vous avez validé l'execution dans votre propre Langfuse --> Switch de clés pour UN SEUL APPEL vers les clés de rendu.

## Partie 2 - Le Chef qui Reflechit 

### 2.1 - Planificateur de menu

Creez une fonction `plan_weekly_menu(constraints: str) -> dict` qui fonctionne en 3 etapes :

1. **Planification** : demandez au LLM de decomposer la tache en etapes (ex : "identifier les contraintes", "choisir les proteines", "equilibrer les repas"). Le LLM doit retourner du JSON.
2. **Execution** : pour chaque etape du plan, faites un appel LLM separe. Passez les resultats precedents en contexte.
3. **Synthese** : un dernier appel combine tous les resultats en un menu de la semaine coherent.

Chaque etape doit etre decoree avec `@observe()` pour creer des spans imbriques dans Langfuse.


### 2.2 - Gestion d'erreur

Ajoutez une gestion d'erreur robuste :
- Si le JSON est invalide implementez un éventuel retry (1 fois max)
- Loguez l'erreur dans Langfuse avec `level="ERROR"`

**Livrables** : 
- menu de la semaine genere en multi-etapes, traces imbriquees visibles dans Langfuse.
- Vos traces sont nomées et taggées avec votre nom de groupe (Groupe XXX, Partie 2)
- Une fois que vous avez validé l'execution dans votre propre Langfuse --> Switch de clés pour UN SEUL APPEL vers les clés de rendu.

## Partie 3 - Evaluation et Qualite


### 3.1 - Creez un dataset de test

Creez un dataset Langfuse nomme `chefbot-menu-eval` contenant **au moins 5 cas de test**. Chaque cas doit avoir :
- **input** : une contrainte (ex : `{"constraints": "repas pour diabetique"}`)
- **expected_output** : des criteres de validation (ex : `{"must_avoid": ["sucre", "pates blanches"], "must_include": ["legumes"], "max_calories_per_meal": 600}`)

Soyez creatifs dans les contraintes : allergies, regimes, budget, nombre de convives, preferences culturelles...

### 3.2 - Evaluateur programmatique

Ecrivez une fonction `rule_evaluator(output: str, expected: dict) -> dict` qui verifie automatiquement :
- Les ingredients interdits ne sont PAS mentionnes (`must_avoid`) → score 0 ou 1
- Les ingredients requis SONT mentionnes (`must_include`) → score proportionnel
- Retourne un dictionnaire de scores

### 3.3 - LLM Juge

Ecrivez un deuxieme evaluateur `llm_judge(question: str, output: str, expected: dict) -> dict` qui utilise un LLM pour noter la reponse sur 3 criteres (chacun de 0.0 a 1.0) :
- **pertinence** : la reponse respecte-t-elle les contraintes ?
- **creativite** : les recettes sont-elles variees et originales ?
- **praticite** : les recettes sont-elles realisables pour un non-professionnel ?

Le juge doit repondre en JSON uniquement.

### 3.4 - Lancez l'experience

Utilisez `get_client().run_experiment()` pour executer votre planificateur sur le dataset, avec vos deux evaluateurs. 

**Livrable** : 
- dataset cree, 2 evaluateurs fonctionnels, resultats d'experience visibles dans Langfuse.


## Partie 4 - ChefBot prend les Outils en Main (~45 min)

*Concepts : tool use, function calling, framework smolagents*

### 4.1 - Definissez 3 outils

Creez les outils suivants (avec des donnees simulees) :

1. **`check_fridge()`** : retourne une liste d'ingredients disponibles dans le frigo

2. **`get_recipe(dish_name: str)`** : retourne une recette detaillee pour un plat donne

3. **`check_dietary_info(ingredient: str)`** : retourne les informations nutritionnelles et allergeniques d'un ingredient


### 4.2 - Boucle de tool calling manuelle

Implementez la boucle de tool calling **sans framework** (comme dans le fichier 06 du cours) :
- Envoyez la question + les definitions d'outils au LLM
- Si le LLM demande un outil, executez-le et renvoyez le resultat
- Continuez jusqu'a une reponse finale
- Maximum 5 iterations


### 4.3 - Migration vers smolagents

Reecrivez la meme logique en utilisant smolagents :
- Utilisez le decorateur `@tool` pour vos 3 outils
- Creez un `CodeAgent` avec `LiteLLMModel`
- Testez avec la meme question

Comparez : quel code est plus court ? Quels avantages apporte le framework ?

**Livrable** : 3 outils, boucle manuelle fonctionnelle, version smolagents fonctionnelle.

## Partie 5 - Le Restaurant Intelligent

### 5.1 - Outil de base de donnees

Creez une classe `MenuDatabaseTool` qui herite de `Tool` (pas le decorateur `@tool`). Cette classe doit :
- S'initialiser avec une base de donnees de plats (nom, prix, temps de preparation, allergenes, categorie)
- Avoir une methode `forward()` qui accepte une recherche par critere (categorie, prix max, sans allergene X...)
- Contenir au moins 10 plats

### 5.2 - Agent avec planification

Creez un agent avec `planning_interval=2` et des custom instructions pour agir comme un serveur de restaurant. L'agent doit :
- Consulter le menu via `MenuDatabaseTool`
- Calculer le total avec un outil `calculate`
- Respecter les contraintes du client

Testez : `"On est 3. Un vegetarien, un sans gluten, et moi je mange de tout. Budget max 60 euros pour le groupe. Proposez-nous un menu complet."`

### 5.3 - Agent conversationnel

Modifiez votre agent pour supporter le mode conversationnel (`reset=False`). Simulez un dialogue de 3 tours :
1. Le client demande des suggestions
2. Le client change d'avis sur un plat
3. Le client demande l'addition

**Livrable** : 
- `MenuDatabaseTool` fonctionnel, agent planificateur, agent conversationnel multi-tours.
- Pour l'aspect tracing comme on a pas vu avec langfuse, ajoutez le dans votre repos Git dans un fichier txt (eg. run_1234.txt)


## Partie 6 - L'Empire ChefBot 

### 6.1 - Architecture multi-agent

Construisez un systeme avec un **manager** et **3 agents specialises** :

| Agent | Role | Outils |
|-------|------|--------|
| `nutritionist` | Verifie l'equilibre nutritionnel et les allergenes | `check_dietary_info` |
| `chef_agent` | Propose des recettes et consulte le frigo | `check_fridge`, `get_recipe` |
| `budget_agent` | Calcule les couts et respecte le budget | `calculate`, `MenuDatabaseTool` |

Le **manager** n'a aucun outil propre. Il recoit la demande du client et delegue aux agents specialises.

### 6.2 - Test du systeme

Testez avec une requete complexe :

```
"Je recois 8 personnes samedi soir. Parmi eux : 2 vegetariens, 1 intolerant au gluten,
1 allergique aux fruits a coque. Budget total : 120 euros.
Je veux un aperitif, une entree, un plat principal et un dessert.
Il faut que tout le monde puisse manger chaque service."
```

**Livrable** : systeme multi-agent fonctionnel, requete complexe traitee avec succes.

## Partie 7 - BOSS FINAL : La Boucle est Bouclee (~30 min)

*Concepts : evaluation end-to-end d'un systeme agentique*

### 7.1 - Dataset de scenarios

Creez un dataset Langfuse `chefbot-multiagent-eval` avec **au moins 4 scenarios** de difficulte croissante :

| # | Scenario | Difficulte |
|---|----------|-----------|
| 1 | Diner simple pour 2 | Facile |
| 2 | Repas famille avec 1 allergie | Moyen |
| 3 | Diner 6 personnes, multi-contraintes, budget serre | Difficile |
| 4 | Evenement 10+ personnes, contraintes culturelles + medicales + budget | Extreme |

Pour chaque scenario, definissez des `expected_output` avec :
- `must_respect` : liste des contraintes a respecter absolument
- `expected_services` : nombre de plats attendus
- `max_budget` : budget a ne pas depasser

### 7.2 - LLM Juge pour systeme multi-agent

Creez un juge LLM qui evalue sur **5 criteres** :
1. **respect_contraintes** : toutes les restrictions alimentaires sont respectees
2. **completude** : tous les services demandes sont proposes
3. **budget** : le budget est respecte
4. **coherence** : le menu forme un ensemble harmonieux
5. **faisabilite** : les recettes sont realisables pour un amateur

### 7.3 - Experimentez et comparez

Lancez l'evaluation et comparez au moins **2 configurations** differentes :
- Changez le modele (ex : `llama-3.3-70b-versatile` vs `llama-4-scout-17b-16e-instruct`)
- OU changez le `planning_interval`
- OU changez les instructions du manager

Analysez : quelle configuration produit les meilleurs resultats ? Pourquoi ?

**Livrable** : dataset de scenarios, juge 5 criteres, comparaison de 2 configurations avec analyse dans Langfuse.

## Bareme indicatif

| Partie | Points | Criteres |
|--------|--------|----------|
| 1 - Premier Contact | 3 | Code fonctionnel, traces Langfuse |
| 2 - Multi-etapes | 2 | Pipeline 3 etapes, spans imbriques, gestion erreur |
| 3 - Evaluation | 3 | Dataset pertinent, 2 evaluateurs, experience executee |
| 4 - Tool Use | 3 | 3 outils, boucle manuelle, version smolagents |
| 5 - Restaurant | 3 | Custom Tool class, planning, conversation multi-tours |
| 6 - Multi-Agent | 3 | Architecture 3 agents + manager, requete complexe |
| 7 - Boss Final | 3 | Dataset scenarios, juge 5 criteres, comparaison |
| **Total** | **20** | |

**Bonus (+2)** : Le projet est propre sur github avec une belle gestion d'environnement.
**Bonus (+ ??)** : Le projet n'est pas entièrement vibe-codé

**Conseils** : 
- Commencez simple, partez des exemples dans les fichiers de cours avant de demander à GPT.
- Vous avez largement le temps d'avoir la moyenne, profitez en pour assimiler toutes les notions vues.

*Bon appetit et bon courage !*
