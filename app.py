"""
app.py
------
SmartRecipe – Flask Web Application

Routes
------
GET  /           → serve the main UI (templates/index.html)
POST /api/recipe → accept ingredient + preference JSON, run the agent,
                   return a JSON recipe response
GET  /api/health → health check endpoint
"""

import os
import sys

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

# Ensure the project root is on the Python path so sibling modules resolve
sys.path.insert(0, os.path.dirname(__file__))

from recipe_agent import RecipeAgent
from granite_client import is_configured

# ── Flask app setup ───────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Initialise the agent once at startup (warms up the TF-IDF index)
agent = RecipeAgent()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main SmartRecipe UI."""
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Simple health-check endpoint."""
    return jsonify({
        "status": "ok",
        "app": "SmartRecipe",
        "granite_configured": is_configured(),
        "mode": "granite" if is_configured() else "demo",
    })


@app.route("/api/recipe", methods=["POST"])
def generate_recipe():
    """
    Accept a JSON body with user preferences and return a generated recipe.

    Expected JSON body
    ------------------
    {
        "ingredients"   : "tomato, onion, paneer, garlic",
        "servings"      : 2,
        "dietary_pref"  : "vegetarian",   // "vegetarian"|"vegan"|"non-vegetarian"|"none"
        "max_time"      : 30,
        "spice_level"   : "medium"        // "mild"|"medium"|"hot"
    }
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON."}), 400

    data = request.get_json(silent=True) or {}

    # ── Extract & validate inputs ─────────────────────────────────────────────
    raw_ingredients = str(data.get("ingredients", "")).strip()
    if not raw_ingredients:
        return jsonify({"success": False, "error": "Please enter at least one ingredient."}), 400

    try:
        servings = max(1, min(20, int(data.get("servings", 2))))
    except (TypeError, ValueError):
        servings = 2

    dietary_pref = str(data.get("dietary_pref", "none")).strip().lower()
    valid_dietary = {"none", "vegetarian", "vegan", "non-vegetarian", "all"}
    if dietary_pref not in valid_dietary:
        dietary_pref = "none"

    try:
        max_time = max(5, min(180, int(data.get("max_time", 60))))
    except (TypeError, ValueError):
        max_time = 60

    spice_level = str(data.get("spice_level", "medium")).strip().lower()
    valid_spice = {"mild", "medium", "hot", "any"}
    if spice_level not in valid_spice:
        spice_level = "medium"

    # ── Run the agent ─────────────────────────────────────────────────────────
    try:
        result = agent.run(
            raw_ingredients=raw_ingredients,
            servings=servings,
            dietary_pref=dietary_pref,
            max_time=max_time,
            spice_level=spice_level,
        )
    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"Agent error: {str(exc)}",
        }), 500

    if not result["success"]:
        # Prefer a specific error from the generation step; fall back to generic
        error_msg = (
            result.get("error")
            or "Could not generate a recipe. Please try again."
        )
        return jsonify({
            "success": False,
            "error": error_msg,
        }), 422

    # ── Build the API response ────────────────────────────────────────────────
    # Summarise agent steps for the UI's "How it worked" panel
    steps_summary = [
        {
            "step": s["step"],
            "tool": s["tool"],
            "status": s["status"],
            "duration_ms": s["duration_ms"],
            "message": s["result"].get("message", ""),
        }
        for s in result.get("steps", [])
    ]

    return jsonify({
        "success": True,
        "recipe_text": result["recipe_text"],
        "retrieved_recipes": result["retrieved_recipes"],
        "ingredients_used": result["ingredients_used"],
        "mode": result["mode"],
        "model": result["model"],
        "is_demo": result["is_demo"],
        "elapsed_seconds": result["elapsed_seconds"],
        "agent_steps": steps_summary,
    })


# ── Entry-point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    print("=" * 60)
    print("  🍳  SmartRecipe – AI Recipe Preparation Agent")
    print("=" * 60)
    print(f"  Mode    : {'IBM Granite (watsonx.ai)' if is_configured() else 'Demo (no credentials)'}")
    print(f"  URL     : http://localhost:{port}")
    print(f"  Debug   : {debug}")
    print("=" * 60)

    app.run(host="0.0.0.0", port=port, debug=debug)
