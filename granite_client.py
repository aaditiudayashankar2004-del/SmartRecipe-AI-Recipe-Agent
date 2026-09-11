"""
granite_client.py
-----------------
IBM Granite integration via IBM watsonx.ai.

All credentials are read from environment variables (never hardcoded).
Configure the following variables in a .env file or your shell:

    WATSONX_API_KEY   – IBM Cloud API key
    WATSONX_PROJECT_ID – watsonx.ai project ID
    WATSONX_URL       – watsonx.ai endpoint (default: us-south)

If credentials are absent the client runs in DEMO MODE and returns a
clearly labelled fallback response so the UI remains functional without
a live API connection.
"""

import os
import textwrap

# ── Try to import the IBM watsonx SDK ─────────────────────────────────────────
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

    IBM_SDK_AVAILABLE = True
except ImportError:
    IBM_SDK_AVAILABLE = False


# ── Configuration ─────────────────────────────────────────────────────────────
WATSONX_URL = os.getenv(
    "WATSONX_URL", "https://us-south.ml.cloud.ibm.com"
)
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")

# IBM Granite model identifier on watsonx.ai
GRANITE_MODEL_ID = "ibm/granite-4-h-small"

# Generation parameters
GENERATION_PARAMS = {
    "max_new_tokens": 1200,
    "min_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}


def is_configured() -> bool:
    """Return True if IBM watsonx.ai credentials are present and the SDK is installed."""
    return (
        IBM_SDK_AVAILABLE
        and bool(WATSONX_API_KEY)
        and bool(WATSONX_PROJECT_ID)
    )


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(
    rag_context: str,
    user_ingredients: list[str],
    servings: int = 2,
    dietary_pref: str = "none",
    max_time: int = 60,
    spice_level: str = "medium",
) -> str:
    """
    Build a structured instruction prompt for IBM Granite.
    The RAG context (retrieved recipes) is embedded directly in the prompt.
    """
    ingredients_str = ", ".join(user_ingredients) if user_ingredients else "assorted ingredients"
    dietary_note = (
        f" The recipe must be {dietary_pref}." if dietary_pref not in ("none", "all", "") else ""
    )

    prompt = textwrap.dedent(f"""
    You are SmartRecipe, an expert AI culinary assistant. Use the retrieved recipe context below to generate a personalised, detailed recipe.

    {rag_context}

    === USER REQUEST ===
    Available ingredients: {ingredients_str}
    Servings needed      : {servings}
    Dietary preference   : {dietary_pref or 'no restriction'}{dietary_note}
    Max cooking time     : {max_time} minutes
    Spice level          : {spice_level}

    === YOUR TASK ===
    Using the recipe context above as inspiration and the user's available ingredients, create a complete personalised recipe.
    Respond ONLY with the following structured format — no extra text before or after:

    ## Recipe Name
    [Creative, descriptive name]

    ## Overview
    [2–3 sentence description of the dish]

    ## Ingredients & Quantities (for {servings} servings)
    [List every ingredient with exact quantity, e.g. "- 200g paneer, cubed"]

    ## Step-by-Step Preparation
    [Numbered steps, clear and detailed]

    ## Cooking Time
    - Prep time: X minutes
    - Cook time: X minutes
    - Total time: X minutes

    ## Servings
    {servings} servings

    ## Substitutions
    [List 3–4 ingredient substitutions for unavailable items]

    ## Cooking Tips
    [2–3 practical tips for best results]

    ## Dietary Adjustments
    [How to make it vegan / gluten-free / lower-calorie if relevant]

    ## Food-Waste Reduction Tips
    [Creative ways to use leftover ingredients or trimmings from this recipe]
    """).strip()

    return prompt


# ── Granite client ────────────────────────────────────────────────────────────

def generate_recipe(prompt: str) -> dict:
    """
    Call IBM Granite on watsonx.ai with the given prompt.

    Returns
    -------
    {
        "success"    : bool,
        "recipe_text": str,
        "mode"       : "granite" | "demo",
        "model"      : str,
    }
    """
    if not is_configured():
        return _demo_response()

    try:
        credentials = Credentials(
            url=WATSONX_URL,
            api_key=WATSONX_API_KEY,
        )

        params = {
            GenParams.MAX_NEW_TOKENS: GENERATION_PARAMS["max_new_tokens"],
            GenParams.MIN_NEW_TOKENS: GENERATION_PARAMS["min_new_tokens"],
            GenParams.TEMPERATURE: GENERATION_PARAMS["temperature"],
            GenParams.TOP_P: GENERATION_PARAMS["top_p"],
            GenParams.REPETITION_PENALTY: GENERATION_PARAMS["repetition_penalty"],
        }

        model = ModelInference(
            model_id=GRANITE_MODEL_ID,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID,
            params=params,
        )

        response = model.generate_text(prompt=prompt)

        return {
            "success": True,
            "recipe_text": response,
            "mode": "granite",
            "model": GRANITE_MODEL_ID,
        }

    except Exception as exc:
        return {
            "success": False,
            "recipe_text": "",
            "mode": "error",
            "model": GRANITE_MODEL_ID,
            "error": f"IBM Granite API error: {str(exc)}",
        }


# ── Demo / fallback mode ──────────────────────────────────────────────────────

def _demo_response() -> dict:
    """
    Return a rich static demo recipe when IBM credentials are not configured.
    Clearly marked so evaluators know it's a fallback.
    """
    recipe_text = textwrap.dedent("""
    > ⚠️ **DEMO MODE** — IBM Granite credentials are not configured.
    > Set WATSONX_API_KEY and WATSONX_PROJECT_ID in your .env file to enable live AI generation.
    > The recipe below is a pre-built demonstration response.

    ---

    ## Recipe Name
    Spiced Paneer & Chickpea Bowl

    ## Overview
    A hearty and wholesome bowl that combines golden-seared paneer cubes with protein-rich chickpeas in a lightly spiced tomato gravy. Ready in under 30 minutes, this dish is satisfying, nutritious and endlessly adaptable to whatever spices you have on hand.

    ## Ingredients & Quantities (for 2 servings)
    - 200g paneer, cut into 2 cm cubes
    - 1 cup cooked chickpeas (or 1 can, drained)
    - 2 medium tomatoes, finely chopped
    - 1 medium onion, finely diced
    - 3 cloves garlic, minced
    - 1 tsp fresh ginger, grated
    - 1 tsp cumin seeds
    - ½ tsp turmeric powder
    - 1 tsp coriander powder
    - ½ tsp red chili powder (adjust to taste)
    - ½ tsp garam masala
    - 2 tbsp oil
    - Salt to taste
    - Fresh cilantro for garnish

    ## Step-by-Step Preparation
    1. Heat 1 tbsp oil in a non-stick pan over medium-high heat. Add paneer cubes and sear for 2 minutes per side until golden. Remove and set aside.
    2. In the same pan, add the remaining oil. Add cumin seeds and let them splutter for 30 seconds.
    3. Add diced onion and sauté for 5–6 minutes until translucent and lightly golden.
    4. Add garlic and ginger; cook for 1 minute until fragrant.
    5. Add chopped tomatoes, turmeric, coriander powder and red chili powder. Cook on medium heat for 6–7 minutes, stirring occasionally, until the oil begins to separate from the masala.
    6. Add chickpeas and ½ cup water. Stir well, cover and simmer for 5 minutes.
    7. Add the seared paneer cubes. Sprinkle garam masala. Stir gently and cook uncovered for 3 minutes.
    8. Taste and adjust salt. Garnish with fresh cilantro and serve.

    ## Cooking Time
    - Prep time: 10 minutes
    - Cook time: 20 minutes
    - Total time: 30 minutes

    ## Servings
    2 servings

    ## Substitutions
    - **Paneer → Tofu**: Press and cube firm tofu for a vegan alternative with similar texture.
    - **Chickpeas → Kidney beans**: Adds an earthier, heartier flavour profile.
    - **Fresh tomatoes → Canned crushed tomatoes**: Use ¾ cup canned for convenience.
    - **Garam masala → Curry powder**: A 1:1 swap works well in a pinch.

    ## Cooking Tips
    - Searing the paneer first creates a golden crust that prevents it from crumbling in the gravy.
    - Let the masala cook until the oil separates — this is the key to a deep, restaurant-style flavour.
    - A small pinch of sugar balances the acidity of the tomatoes beautifully.

    ## Dietary Adjustments
    - **Vegan**: Replace paneer with firm tofu and use neutral oil instead of any ghee.
    - **Gluten-free**: This recipe is naturally gluten-free — just verify your spice blends contain no fillers.
    - **Lower calorie**: Reduce oil to 1 tsp and use an air-fryer to crisp the paneer instead of pan-frying.

    ## Food-Waste Reduction Tips
    - Use the chickpea liquid (aquafaba) as an egg replacer in baking or to thicken soups.
    - Onion and tomato scraps can be simmered into a simple vegetable stock.
    - Leftover paneer-chickpea curry makes an excellent stuffing for wraps or sandwiches the next day.
    """).strip()

    return {
        "success": True,
        "recipe_text": recipe_text,
        "mode": "demo",
        "model": "demo (no IBM credentials configured)",
    }
