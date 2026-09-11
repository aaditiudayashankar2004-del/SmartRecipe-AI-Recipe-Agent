"""
rag_engine.py
-------------
Retrieval-Augmented Generation engine for SmartRecipe.

Uses TF-IDF (scikit-learn) to find the most relevant recipes from
recipes.json based on the ingredients the user currently has.
"""

import json
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Path to the recipe knowledge base ────────────────────────────────────────
RECIPES_PATH = os.path.join(os.path.dirname(__file__), "recipes.json")


def _load_recipes() -> list[dict]:
    """Load all recipes from the JSON knowledge base."""
    with open(RECIPES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_recipe_document(recipe: dict) -> str:
    """
    Convert a recipe dict into a single text document used for TF-IDF indexing.
    We combine the name, category, ingredients, tags and description so that
    the retriever can match on any of those dimensions.
    """
    parts = [
        recipe.get("name", ""),
        recipe.get("category", ""),
        recipe.get("description", ""),
        " ".join(recipe.get("ingredients", [])),
        " ".join(recipe.get("tags", [])),
        " ".join(recipe.get("dietary", [])),
    ]
    return " ".join(p for p in parts if p)


class RAGEngine:
    """
    Lightweight RAG retrieval engine backed by TF-IDF cosine similarity.

    Usage
    -----
    engine = RAGEngine()
    results = engine.retrieve(ingredients=["tomato","onion","paneer"], top_k=3)
    """

    def __init__(self):
        self.recipes: list[dict] = _load_recipes()
        self._build_index()

    # ── Index construction ────────────────────────────────────────────────────

    def _build_index(self):
        """Fit a TF-IDF matrix over all recipe documents."""
        documents = [_build_recipe_document(r) for r in self.recipes]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            stop_words="english",
            min_df=1,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        ingredients: list[str],
        top_k: int = 3,
        dietary_filter: str | None = None,
        max_time_minutes: int | None = None,
        spice_level: str | None = None,
    ) -> list[dict]:
        """
        Retrieve the top-k most relevant recipes for the supplied ingredients.

        Parameters
        ----------
        ingredients      : list of ingredient strings the user has available
        top_k            : number of results to return
        dietary_filter   : optional filter – "vegetarian", "vegan", "non-vegetarian"
        max_time_minutes : optional upper bound on cooking time
        spice_level      : optional filter – "mild", "medium", "hot"

        Returns
        -------
        List of recipe dicts, sorted by relevance (highest first), including
        an extra "relevance_score" field (0-1).
        """
        if not ingredients:
            return []

        # Build a query string from the user's ingredients
        query = " ".join(ingredients)

        # TF-IDF similarity
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Attach scores to recipes
        scored = [
            {**recipe, "relevance_score": float(round(scores[i], 4))}
            for i, recipe in enumerate(self.recipes)
        ]

        # ── Filters ───────────────────────────────────────────────────────────

        if dietary_filter and dietary_filter.lower() not in ("all", "any", "none", ""):
            df = dietary_filter.lower()
            scored = [r for r in scored if df in [d.lower() for d in r.get("dietary", [])]]

        if max_time_minutes:
            scored = [r for r in scored if r.get("cooking_time_minutes", 999) <= max_time_minutes]

        if spice_level and spice_level.lower() not in ("any", "all", ""):
            scored = [r for r in scored if r.get("spice_level", "").lower() == spice_level.lower()]

        # ── Sort & return top-k ───────────────────────────────────────────────
        scored.sort(key=lambda r: r["relevance_score"], reverse=True)
        return scored[:top_k]

    # ── Context builder ───────────────────────────────────────────────────────

    def build_rag_context(
        self,
        recipes: list[dict],
        user_ingredients: list[str],
    ) -> str:
        """
        Compose a structured RAG context string from retrieved recipes.
        This is injected into the Granite prompt as supporting knowledge.
        """
        if not recipes:
            return "No matching recipes found in the knowledge base."

        lines = [
            "=== RETRIEVED RECIPE CONTEXT ===",
            f"User has these ingredients: {', '.join(user_ingredients)}",
            "",
        ]
        for i, r in enumerate(recipes, 1):
            lines += [
                f"--- Recipe {i}: {r['name']} ---",
                f"Category   : {r['category']}",
                f"Dietary    : {', '.join(r.get('dietary', []))}",
                f"Cook Time  : {r.get('cooking_time_minutes', '?')} minutes",
                f"Servings   : {r.get('servings', '?')}",
                f"Spice Level: {r.get('spice_level', '?')}",
                f"Ingredients: {', '.join(r.get('ingredients', []))}",
                f"Description: {r.get('description', '')}",
                f"Tags       : {', '.join(r.get('tags', []))}",
                f"Relevance  : {r.get('relevance_score', 0):.2%}",
                "",
            ]

        lines.append("=== END OF CONTEXT ===")
        return "\n".join(lines)


# ── Module-level singleton ────────────────────────────────────────────────────
_engine_instance: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    """Return a cached RAGEngine instance (lazy-initialised)."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine()
    return _engine_instance
