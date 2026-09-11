# 🍳 SmartRecipe – AI Recipe Preparation Agent

> **AICTE Problem Statement 16 – Recipe Preparation Agent**  
> An agentic, RAG-based recipe assistant powered by IBM Granite (watsonx.ai)

---

## What is SmartRecipe?

SmartRecipe is a full-stack AI web application that turns a list of ingredients into a personalised, step-by-step recipe. It combines:

- **Retrieval-Augmented Generation (RAG)** – finds the most relevant recipes from a local knowledge base using TF-IDF cosine similarity
- **IBM Granite on watsonx.ai** – generates a detailed, personalised recipe from the retrieved context
- **Agentic AI architecture** – a 5-step tool pipeline orchestrated by `RecipeAgent`
- **Modern Web UI** – responsive, clean Flask + HTML/CSS/JS frontend

---

## Architecture

```
User → Web UI → Flask API → RecipeAgent → 5-step Tool Pipeline
                                │
              ┌─────────────────┼─────────────────────────┐
              │                 │                          │
         ParseIngredients  RetrieveRecipes          BuildContext
              │            (TF-IDF / RAG)                 │
              └─────────────────┼─────────────────────────┘
                                │
                        GenerateRecipe
                       (IBM Granite / Demo)
                                │
                        FormatResponse
                                │
                          Web UI Result
```

---

## File Structure

```
SmartRecipe/
├── app.py               ← Flask web application (routes & API)
├── recipe_agent.py      ← Agentic orchestrator (5-step pipeline)
├── recipe_tools.py      ← Individual agent tools
├── rag_engine.py        ← TF-IDF retrieval engine
├── granite_client.py    ← IBM Granite / watsonx.ai integration ⭐
├── recipes.json         ← Local recipe knowledge base (15 recipes)
├── requirements.txt     ← Python dependencies
├── .env.example         ← Environment variable template
├── .gitignore
├── templates/
│   └── index.html       ← Main web UI
└── static/
    ├── style.css        ← Responsive styling
    └── script.js        ← Frontend logic & Markdown renderer
```

> ⭐ **IBM Granite configuration is in `granite_client.py`**

---

## Quick Start

### 1. Clone / download the project

```bash
cd SmartRecipe
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure IBM Granite credentials (optional)

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
WATSONX_API_KEY=your_ibm_cloud_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

> **Don't have credentials?** Leave both fields blank — the app will run in **Demo Mode** and still work fully for demonstration purposes.

### 5. Run the application

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## How to Get IBM watsonx.ai Credentials

1. Sign up at [https://cloud.ibm.com](https://cloud.ibm.com) (free tier available)
2. Create a **watsonx.ai** service instance
3. Create a **Project** at [https://dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com)
4. Copy your **Project ID** from the project settings
5. Generate an **API key** at [https://cloud.ibm.com/iam/apikeys](https://cloud.ibm.com/iam/apikeys)
6. Add both values to your `.env` file

---

## IBM Granite Configuration

All IBM integration lives in **`granite_client.py`**:

| Variable | Description |
|---|---|
| `WATSONX_API_KEY` | Your IBM Cloud API key |
| `WATSONX_PROJECT_ID` | Your watsonx.ai project ID |
| `WATSONX_URL` | Regional endpoint (default: `us-south`) |
| `GRANITE_MODEL_ID` | Model used: `ibm/granite-3-3-8b-instruct` |

The `is_configured()` function checks whether credentials are present. If they are missing, `granite_client.py` automatically returns a clearly-labelled demo response.

---

## Agent Tool Pipeline

| Step | Tool | What it does |
|---|---|---|
| 1 | `ParseIngredients` | Normalises raw ingredient text (splits, lowercases, deduplicates) |
| 2 | `RetrieveRecipes` | TF-IDF cosine similarity search over `recipes.json` with dietary/time/spice filters |
| 3 | `BuildContext` | Formats retrieved recipes into a structured RAG context string |
| 4 | `GenerateRecipe` | Sends context + user preferences to IBM Granite; falls back to demo if unconfigured |
| 5 | `FormatResponse` | Assembles the final JSON response with metadata for the UI |

---

## Recipe Knowledge Base

`recipes.json` contains **15 recipes** covering:

| Category | Examples |
|---|---|
| Indian Vegetarian | Paneer Butter Masala, Dal Tadka, Palak Paneer, Aloo Gobi, Rajma Chawal |
| Indian Non-Vegetarian | Chicken Tikka Masala, Chicken Biryani, Mutton Rogan Josh |
| Indian Vegan | Chana Masala |
| Quick Meals (≤20 min) | Egg Fried Rice, Masoor Dal Soup, Tofu Stir Fry, Omelette Wrap, Vegan Buddha Bowl |

---

## What the Generated Recipe Includes

- Recipe name and overview
- Ingredients with exact quantities (scaled to servings)
- Numbered step-by-step preparation
- Prep + cook + total time
- Servings
- Ingredient substitutions
- Cooking tips
- Dietary adjustments (vegan/gluten-free/lower-calorie)
- Food-waste reduction suggestions

---

## Demo Mode

When IBM credentials are **not** configured:
- The app runs fully in the browser
- A clearly-labelled demo recipe is returned
- A yellow banner in the UI explains demo mode
- The agent still runs all 5 steps (only Step 4 uses the static response)

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Flask | Web framework |
| scikit-learn | TF-IDF vectoriser for RAG retrieval |
| pandas / numpy | Data utilities |
| ibm-watsonx-ai | IBM Granite SDK |
| python-dotenv | Environment variable management |
| HTML5 / CSS3 | Responsive UI |
| Vanilla JavaScript | Frontend logic |

---

## License

This project is built for the AICTE Problem Statement 16 hackathon submission. All recipe data is fictional and for demonstration purposes only.
