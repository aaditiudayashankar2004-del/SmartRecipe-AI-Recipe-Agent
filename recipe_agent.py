"""
recipe_agent.py
---------------
SmartRecipe – Agentic Orchestrator

The RecipeAgent executes a multi-step tool pipeline to turn a user's
ingredient list into a personalised AI-generated recipe.

Agent Flow
----------
Step 1 │ ParseIngredients  – normalise the raw input text
Step 2 │ RetrieveRecipes   – RAG retrieval from recipes.json (TF-IDF)
Step 3 │ BuildContext      – assemble the RAG context string
Step 4 │ GenerateRecipe    – call IBM Granite (or demo fallback)
Step 5 │ FormatResponse    – package the final structured response

Each step is a discrete tool call (see recipe_tools.py). The agent
records every step in an execution trace so the caller can see exactly
what the agent did and why.
"""

import time
from typing import Any

from recipe_tools import (
    parse_ingredients_tool,
    retrieve_recipes_tool,
    build_context_tool,
    generate_recipe_tool,
    format_response_tool,
)


class RecipeAgent:
    """
    Agentic orchestrator for SmartRecipe.

    Usage
    -----
    agent = RecipeAgent()
    result = agent.run(
        raw_ingredients="tomato, onion, paneer, garlic",
        servings=2,
        dietary_pref="vegetarian",
        max_time=30,
        spice_level="medium",
    )
    """

    def __init__(self):
        self.name = "SmartRecipe Agent"
        self.version = "1.0.0"

    # ── Public entry-point ────────────────────────────────────────────────────

    def run(
        self,
        raw_ingredients: str,
        servings: int = 2,
        dietary_pref: str = "none",
        max_time: int = 60,
        spice_level: str = "medium",
        top_k: int = 3,
    ) -> dict[str, Any]:
        """
        Execute the full agent pipeline and return a structured result.

        Parameters
        ----------
        raw_ingredients : comma-separated ingredients the user has
        servings        : number of servings to generate recipe for
        dietary_pref    : "vegetarian" | "vegan" | "non-vegetarian" | "none"
        max_time        : maximum cooking time in minutes
        spice_level     : "mild" | "medium" | "hot"
        top_k           : number of recipes to retrieve from the knowledge base

        Returns
        -------
        {
            "agent_name"       : str,
            "success"          : bool,
            "steps"            : list[dict],   # full execution trace
            "recipe_text"      : str,
            "retrieved_recipes": list[dict],
            "ingredients_used" : list[str],
            "mode"             : str,
            "model"            : str,
            "elapsed_seconds"  : float,
            "is_demo"          : bool,
            "error"            : str | None,
        }
        """
        start_time = time.time()
        steps: list[dict] = []

        # ── Step 1: Parse Ingredients ─────────────────────────────────────────
        step1 = self._run_step(
            step_number=1,
            tool_name="ParseIngredients",
            description="Normalise and validate the raw ingredient input.",
            fn=lambda: parse_ingredients_tool(raw_ingredients),
        )
        steps.append(step1)

        if not step1["result"]["success"]:
            return self._error_response(
                steps=steps,
                message=step1["result"]["message"],
                elapsed=time.time() - start_time,
            )

        ingredients = step1["result"]["ingredients"]

        # ── Step 2: Retrieve Recipes (RAG) ────────────────────────────────────
        # Normalise filter values so empty strings behave as "no filter"
        dietary_arg = dietary_pref if dietary_pref not in ("none", "") else None
        spice_arg = spice_level if spice_level not in ("any", "") else None
        time_arg = max_time if max_time > 0 else None

        step2 = self._run_step(
            step_number=2,
            tool_name="RetrieveRecipes",
            description="Search the recipe knowledge base using TF-IDF cosine similarity.",
            fn=lambda: retrieve_recipes_tool(
                ingredients=ingredients,
                top_k=top_k,
                dietary_filter=dietary_arg,
                max_time_minutes=time_arg,
                spice_level=spice_arg,
            ),
        )
        steps.append(step2)

        # If retrieval fails with filters, retry without filters
        if not step2["result"]["success"]:
            step2_retry = self._run_step(
                step_number="2b",
                tool_name="RetrieveRecipes (retry without filters)",
                description="Retrying retrieval with filters relaxed.",
                fn=lambda: retrieve_recipes_tool(
                    ingredients=ingredients,
                    top_k=top_k,
                ),
            )
            steps.append(step2_retry)
            retrieval_result = step2_retry["result"]
        else:
            retrieval_result = step2["result"]

        # ── Step 3: Build RAG Context ─────────────────────────────────────────
        step3 = self._run_step(
            step_number=3,
            tool_name="BuildContext",
            description="Assemble the RAG context from retrieved recipes.",
            fn=lambda: build_context_tool(
                recipes=retrieval_result.get("recipes", []),
                user_ingredients=ingredients,
            ),
        )
        steps.append(step3)
        rag_context = step3["result"]["context"]

        # ── Step 4: Generate Recipe via IBM Granite ───────────────────────────
        step4 = self._run_step(
            step_number=4,
            tool_name="GenerateRecipe",
            description="Send RAG context to IBM Granite (watsonx.ai) and generate recipe.",
            fn=lambda: generate_recipe_tool(
                rag_context=rag_context,
                user_ingredients=ingredients,
                servings=servings,
                dietary_pref=dietary_pref,
                max_time=max_time,
                spice_level=spice_level,
            ),
        )
        steps.append(step4)

        # ── Step 5: Format Final Response ─────────────────────────────────────
        elapsed = time.time() - start_time
        step5 = self._run_step(
            step_number=5,
            tool_name="FormatResponse",
            description="Package the generated recipe and metadata into the final response.",
            fn=lambda: format_response_tool(
                generation_result=step4["result"],
                retrieval_result=retrieval_result,
                parse_result=step1["result"],
                elapsed_seconds=elapsed,
            ),
        )
        steps.append(step5)

        final = step5["result"]
        return {
            "agent_name": self.name,
            "success": final["success"],
            "steps": steps,
            "recipe_text": final.get("recipe_text", ""),
            "retrieved_recipes": final.get("retrieved_recipes", []),
            "ingredients_used": final.get("ingredients_used", []),
            "mode": final.get("mode", "unknown"),
            "model": final.get("model", ""),
            "elapsed_seconds": final.get("elapsed_seconds", round(elapsed, 2)),
            "is_demo": final.get("is_demo", False),
            "error": final.get("error") or None,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_step(
        self,
        step_number: int | str,
        tool_name: str,
        description: str,
        fn,
    ) -> dict[str, Any]:
        """
        Execute a single tool call and wrap it in a step record.
        Catches unexpected exceptions so one bad tool cannot crash the agent.
        """
        step_start = time.time()
        try:
            result = fn()
            status = "success" if result.get("success", True) else "failed"
        except Exception as exc:
            result = {"success": False, "message": str(exc)}
            status = "error"

        return {
            "step": step_number,
            "tool": tool_name,
            "description": description,
            "status": status,
            "duration_ms": round((time.time() - step_start) * 1000),
            "result": result,
        }

    def _error_response(
        self,
        steps: list[dict],
        message: str,
        elapsed: float,
    ) -> dict[str, Any]:
        return {
            "agent_name": self.name,
            "success": False,
            "steps": steps,
            "recipe_text": "",
            "retrieved_recipes": [],
            "ingredients_used": [],
            "mode": "error",
            "model": "",
            "elapsed_seconds": round(elapsed, 2),
            "is_demo": False,
            "error": message,
        }
