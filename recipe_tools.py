"""
recipe_tools.py
---------------
Individual agent tools used by the RecipeAgent.

Each tool is a standalone callable that performs ONE focused task.
The agent orchestrates these tools in sequence to produce a final result.

Tools
-----
1. parse_ingredients_tool  – normalise raw ingredient text from the user
2. retrieve_recipes_tool   – search the RAG engine for relevant recipes
3. build_context_tool      – assemble the RAG context string
4. generate_recipe_tool    – call IBM Granite with the RAG context
5. format_response_tool    – add metadata and clean up the response
"""

import re
import time
from typing import Any

from rag_engine import get_rag_engine
from granite_client import build_prompt, generate_recipe, is_configured


# ── Tool 1: Parse & normalise ingredients ────────────────────────────────────

def parse_ingredients_tool(raw_input: str) -> dict[str, Any]:
    """
    Tool: ParseIngredients
    ----------------------
    Accepts a comma/newline/semicolon-separated string of ingredient names
    and returns a clean list of normalised strings.

    Returns
    -------
    {
        "tool": "ParseIngredients",
        "success": bool,
        "ingredients": list[str],
        "count": int,
        "message": str,
    }
    """
    if not raw_input or not raw_input.strip():
        return {
            "tool": "ParseIngredients",
            "success": False,
            "ingredients": [],
            "count": 0,
            "message": "No ingredients provided. Please enter at least one ingredient.",
        }

    # Split on comma, semicolon, or newline
    raw_items = re.split(r"[,;\n]+", raw_input)

    # Normalise: lowercase, strip whitespace, remove empty strings
    cleaned = [item.strip().lower() for item in raw_items if item.strip()]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for item in cleaned:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    if not unique:
        return {
            "tool": "ParseIngredients",
            "success": False,
            "ingredients": [],
            "count": 0,
            "message": "Could not parse any ingredients. Please separate them with commas.",
        }

    return {
        "tool": "ParseIngredients",
        "success": True,
        "ingredients": unique,
        "count": len(unique),
        "message": f"Parsed {len(unique)} ingredient(s): {', '.join(unique)}.",
    }


# ── Tool 2: Retrieve relevant recipes via RAG ─────────────────────────────────

def retrieve_recipes_tool(
    ingredients: list[str],
    top_k: int = 3,
    dietary_filter: str | None = None,
    max_time_minutes: int | None = None,
    spice_level: str | None = None,
) -> dict[str, Any]:
    """
    Tool: RetrieveRecipes
    ---------------------
    Queries the RAG engine (TF-IDF) with the user's ingredients and optional
    filters to retrieve the most relevant recipes from the knowledge base.

    Returns
    -------
    {
        "tool": "RetrieveRecipes",
        "success": bool,
        "recipes": list[dict],
        "count": int,
        "message": str,
    }
    """
    try:
        engine = get_rag_engine()
        results = engine.retrieve(
            ingredients=ingredients,
            top_k=top_k,
            dietary_filter=dietary_filter,
            max_time_minutes=max_time_minutes,
            spice_level=spice_level,
        )

        if not results:
            return {
                "tool": "RetrieveRecipes",
                "success": False,
                "recipes": [],
                "count": 0,
                "message": (
                    "No recipes matched the given filters. "
                    "Try removing dietary or spice restrictions and searching again."
                ),
            }

        names = [r["name"] for r in results]
        return {
            "tool": "RetrieveRecipes",
            "success": True,
            "recipes": results,
            "count": len(results),
            "message": f"Found {len(results)} relevant recipe(s): {', '.join(names)}.",
        }

    except Exception as exc:
        return {
            "tool": "RetrieveRecipes",
            "success": False,
            "recipes": [],
            "count": 0,
            "message": f"Retrieval error: {str(exc)}",
        }


# ── Tool 3: Build the RAG context string ─────────────────────────────────────

def build_context_tool(
    recipes: list[dict],
    user_ingredients: list[str],
) -> dict[str, Any]:
    """
    Tool: BuildContext
    ------------------
    Converts retrieved recipes into a structured RAG context string that
    is injected into the Granite prompt.

    Returns
    -------
    {
        "tool": "BuildContext",
        "success": bool,
        "context": str,
        "message": str,
    }
    """
    try:
        engine = get_rag_engine()
        context = engine.build_rag_context(recipes=recipes, user_ingredients=user_ingredients)
        return {
            "tool": "BuildContext",
            "success": True,
            "context": context,
            "message": f"Built RAG context from {len(recipes)} recipe(s).",
        }
    except Exception as exc:
        return {
            "tool": "BuildContext",
            "success": False,
            "context": "No context available.",
            "message": f"Context building error: {str(exc)}",
        }


# ── Tool 4: Generate recipe via IBM Granite ───────────────────────────────────

def generate_recipe_tool(
    rag_context: str,
    user_ingredients: list[str],
    servings: int = 2,
    dietary_pref: str = "none",
    max_time: int = 60,
    spice_level: str = "medium",
) -> dict[str, Any]:
    """
    Tool: GenerateRecipe
    --------------------
    Builds the Granite prompt from the RAG context and user preferences,
    then calls IBM Granite (or the demo fallback) to generate the recipe.

    Returns
    -------
    {
        "tool": "GenerateRecipe",
        "success": bool,
        "recipe_text": str,
        "mode": str,
        "model": str,
        "message": str,
    }
    """
    try:
        prompt = build_prompt(
            rag_context=rag_context,
            user_ingredients=user_ingredients,
            servings=servings,
            dietary_pref=dietary_pref,
            max_time=max_time,
            spice_level=spice_level,
        )

        result = generate_recipe(prompt=prompt)

        mode_label = {
            "granite": "IBM Granite (watsonx.ai)",
            "demo": "Demo Mode (no credentials configured)",
            "error": "Error fallback",
        }.get(result.get("mode", ""), result.get("mode", ""))

        return {
            "tool": "GenerateRecipe",
            "success": result["success"],
            "recipe_text": result["recipe_text"],
            "mode": result["mode"],
            "model": result["model"],
            "message": f"Recipe generated via {mode_label}.",
        }

    except Exception as exc:
        return {
            "tool": "GenerateRecipe",
            "success": False,
            "recipe_text": "",
            "mode": "error",
            "model": "unknown",
            "message": f"Generation error: {str(exc)}",
        }


# ── Tool 5: Format the final response ─────────────────────────────────────────

def format_response_tool(
    generation_result: dict[str, Any],
    retrieval_result: dict[str, Any],
    parse_result: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """
    Tool: FormatResponse
    --------------------
    Assembles the final structured response that is sent back to the
    Flask API and then rendered in the UI.

    Returns
    -------
    {
        "tool": "FormatResponse",
        "success": bool,
        "recipe_text": str,
        "retrieved_recipes": list[dict],
        "ingredients_used": list[str],
        "mode": str,
        "model": str,
        "elapsed_seconds": float,
        "is_demo": bool,
        "message": str,
    }
    """
    recipe_text = generation_result.get("recipe_text", "")
    mode = generation_result.get("mode", "unknown")
    gen_error = generation_result.get("error", "")

    # Summarise retrieved recipes for UI display (name + relevance only)
    retrieved_summary = [
        {
            "name": r.get("name", ""),
            "category": r.get("category", ""),
            "relevance": f"{r.get('relevance_score', 0):.0%}",
            "cook_time": r.get("cooking_time_minutes", "?"),
            "dietary": r.get("dietary", []),
        }
        for r in retrieval_result.get("recipes", [])
    ]

    success = bool(recipe_text)
    return {
        "tool": "FormatResponse",
        "success": success,
        "recipe_text": recipe_text,
        "retrieved_recipes": retrieved_summary,
        "ingredients_used": parse_result.get("ingredients", []),
        "mode": mode,
        "model": generation_result.get("model", ""),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "is_demo": mode == "demo",
        "message": generation_result.get("message", ""),
        "error": gen_error if not success else "",
    }
