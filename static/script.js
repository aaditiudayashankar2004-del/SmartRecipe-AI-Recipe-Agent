/**
 * script.js – SmartRecipe frontend logic
 *
 * Responsibilities
 * ────────────────
 * 1. Check API health on load and update the mode badge
 * 2. Handle quick-fill chips
 * 3. Submit the recipe form via fetch (POST /api/recipe)
 * 4. Animate a live step-tracker during loading
 * 5. Render the result: retrieved recipes, generated recipe text,
 *    agent execution trace
 * 6. Minimal Markdown renderer (no external library required)
 */

"use strict";

// ── DOM references ─────────────────────────────────────────────────────────

const form         = document.getElementById("recipe-form");
const submitBtn    = document.getElementById("submit-btn");
const btnText      = submitBtn.querySelector(".btn-text");
const btnLoading   = submitBtn.querySelector(".btn-loading");

const idleState    = document.getElementById("idle-state");
const loadingState = document.getElementById("loading-state");
const errorState   = document.getElementById("error-state");
const resultState  = document.getElementById("result-state");

const stepTracker  = document.getElementById("step-tracker");
const errorMsg     = document.getElementById("error-message");

const modeBadge    = document.getElementById("mode-badge");

// Result elements
const metaModel    = document.getElementById("meta-model");
const metaMode     = document.getElementById("meta-mode");
const metaTime     = document.getElementById("meta-time");
const demoBanner   = document.getElementById("demo-banner");
const retrievedEl  = document.getElementById("retrieved-recipes");
const recipeEl     = document.getElementById("recipe-content");
const agentStepsEl = document.getElementById("agent-steps");


// ── Agent step labels (for the live tracker animation) ────────────────────

const STEP_LABELS = [
  { icon: "🔤", label: "Parsing your ingredients…" },
  { icon: "🔍", label: "Searching the recipe knowledge base…" },
  { icon: "📋", label: "Building RAG context…" },
  { icon: "🧠", label: "IBM Granite is generating your recipe…" },
  { icon: "✅", label: "Formatting the final response…" },
];


// ── Health check on page load ─────────────────────────────────────────────

async function checkHealth() {
  try {
    const res  = await fetch("/api/health");
    const data = await res.json();
    const isGranite = data.granite_configured;

    modeBadge.textContent = isGranite ? "✓ IBM Granite Active" : "Demo Mode";
    modeBadge.className   = `badge badge--${isGranite ? "granite" : "demo"}`;
  } catch {
    modeBadge.textContent = "⚠ Offline";
    modeBadge.className   = "badge badge--error";
  }
}

checkHealth();


// ── Quick-fill chips ──────────────────────────────────────────────────────

document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    document.getElementById("ingredients").value = chip.dataset.fill;
  });
});


// ── Form submission ───────────────────────────────────────────────────────

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const ingredients = document.getElementById("ingredients").value.trim();
  if (!ingredients) {
    showError("Please enter at least one ingredient.");
    return;
  }

  const payload = {
    ingredients,
    servings:      parseInt(document.getElementById("servings").value,     10),
    dietary_pref:  document.getElementById("dietary_pref").value,
    max_time:      parseInt(document.getElementById("max_time").value,     10),
    spice_level:   document.getElementById("spice_level").value,
  };

  showLoading();

  try {
    const res  = await fetch("/api/recipe", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok || !data.success) {
      showError(data.error || "An unexpected error occurred. Please try again.");
      return;
    }

    showResult(data);

  } catch (err) {
    showError("Network error – could not reach the server. Is Flask running?");
  }
});


// ── UI state transitions ──────────────────────────────────────────────────

/** Switch to the loading state and run the animated step ticker. */
function showLoading() {
  idleState.hidden   = true;
  errorState.hidden  = true;
  resultState.hidden = true;
  loadingState.hidden = false;

  submitBtn.disabled = true;
  btnText.hidden     = true;
  btnLoading.hidden  = false;

  stepTracker.innerHTML = "";
  animateSteps();
}

/** Switch to the error state. */
function showError(message) {
  idleState.hidden    = true;
  loadingState.hidden = true;
  resultState.hidden  = true;
  errorState.hidden   = false;

  errorMsg.textContent = message;
  resetBtn();
}

/** Switch to the result state and render the data. */
function showResult(data) {
  loadingState.hidden = true;
  errorState.hidden   = true;
  idleState.hidden    = true;
  resultState.hidden  = false;

  renderMeta(data);
  renderRetrievedRecipes(data.retrieved_recipes || []);
  renderRecipeText(data.recipe_text || "");
  renderAgentSteps(data.agent_steps || []);

  resetBtn();
}

/** Reset the form button to its default state. */
function resetBtn() {
  submitBtn.disabled = false;
  btnText.hidden     = false;
  btnLoading.hidden  = true;
}

/** Reset the entire UI back to the idle state. */
function resetUI() {
  idleState.hidden    = false;
  loadingState.hidden = true;
  errorState.hidden   = true;
  resultState.hidden  = true;
  resetBtn();
}


// ── Loading step animation ────────────────────────────────────────────────

function animateSteps() {
  stepTracker.innerHTML = "";
  let i = 0;

  function addNextStep() {
    if (i >= STEP_LABELS.length) return;
    const { icon, label } = STEP_LABELS[i];
    const div = document.createElement("div");
    div.className = "tracker-item running";
    div.innerHTML = `<span>${icon}</span><span>${label}</span>`;
    stepTracker.appendChild(div);

    // Mark previous step as done
    if (i > 0) {
      const prev = stepTracker.children[i - 1];
      if (prev) {
        prev.className  = "tracker-item done";
        prev.innerHTML  = prev.innerHTML.replace(prev.children[0].textContent, "✔");
      }
    }
    i++;
    // Stagger steps: faster for first 3, pause on "IBM Granite is generating…"
    const delay = i === 4 ? 1800 : 700;
    setTimeout(addNextStep, delay);
  }

  addNextStep();
}


// ── Result renderers ──────────────────────────────────────────────────────

function renderMeta(data) {
  metaModel.textContent = data.model || "Unknown";
  metaTime.textContent  = `${data.elapsed_seconds}s`;

  metaMode.textContent = {
    granite: "IBM Granite",
    demo:    "Demo Mode",
    error:   "Error Fallback",
  }[data.mode] || data.mode;

  metaMode.className = `meta-value badge badge--${
    data.mode === "granite" ? "granite" :
    data.mode === "demo"    ? "demo"    : "error"
  }`;

  demoBanner.hidden = !data.is_demo;
}

function renderRetrievedRecipes(recipes) {
  retrievedEl.innerHTML = "";

  if (!recipes.length) {
    retrievedEl.innerHTML = '<p style="color:var(--color-muted);font-size:.85rem">No recipes retrieved.</p>';
    return;
  }

  recipes.forEach(r => {
    const pct = parseInt(r.relevance, 10) || 0;
    const dietaryBadges = (r.dietary || [])
      .map(d => `<span class="chip" style="cursor:default;font-size:.7rem;padding:2px 7px">${d}</span>`)
      .join(" ");

    const card = document.createElement("div");
    card.className = "retrieved-card";
    card.innerHTML = `
      <div class="retrieved-name">${escHtml(r.name)}</div>
      <div class="retrieved-meta">
        <div>📂 ${escHtml(r.category)}</div>
        <div>⏱ ${r.cook_time} min</div>
        <div style="margin-top:4px">${dietaryBadges}</div>
      </div>
      <div class="relevance-bar">
        <div class="relevance-fill" style="width:${pct}%"></div>
      </div>
      <div style="font-size:.7rem;color:var(--color-muted);margin-top:3px">
        Match: ${r.relevance}
      </div>
    `;
    retrievedEl.appendChild(card);
  });
}

function renderRecipeText(text) {
  recipeEl.innerHTML = simpleMarkdown(text);
}

function renderAgentSteps(steps) {
  agentStepsEl.innerHTML = "";

  steps.forEach(s => {
    const isOk  = s.status === "success";
    const icon  = isOk ? "✅" : s.status === "failed" ? "⚠️" : "❌";

    const div = document.createElement("div");
    div.className = "agent-step";
    div.innerHTML = `
      <div class="step-icon">${icon}</div>
      <div class="step-info">
        <div class="step-tool">Step ${s.step}: ${escHtml(s.tool)}</div>
        <div class="step-msg">${escHtml(s.message || "")}</div>
      </div>
      <div class="step-duration">${s.duration_ms}ms</div>
    `;
    agentStepsEl.appendChild(div);
  });
}


// ── Minimal Markdown renderer ─────────────────────────────────────────────
// Supports: ## headings, **bold**, *italic*, - lists, 1. lists,
//           > blockquotes, `code`, horizontal rules, line breaks.
// No external dependency required.

function simpleMarkdown(md) {
  if (!md) return "";

  const lines  = md.split("\n");
  const output = [];
  let inUl     = false;
  let inOl     = false;

  function closeList() {
    if (inUl) { output.push("</ul>"); inUl = false; }
    if (inOl) { output.push("</ol>"); inOl = false; }
  }

  function inlineFormat(text) {
    return text
      .replace(/&/g,  "&amp;")
      .replace(/</g,  "&lt;")
      .replace(/>/g,  "&gt;")
      // Bold+italic
      .replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>")
      // Bold
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      // Italic
      .replace(/\*(.+?)\*/g, "<em>$1</em>")
      // Inline code
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  for (let raw of lines) {
    const line = raw.trimEnd();

    // ── Blank line ──
    if (!line.trim()) {
      closeList();
      output.push("<br>");
      continue;
    }

    // ── Headings ──
    if (/^## /.test(line)) {
      closeList();
      output.push(`<h2>${inlineFormat(line.slice(3))}</h2>`);
      continue;
    }
    if (/^### /.test(line)) {
      closeList();
      output.push(`<h3>${inlineFormat(line.slice(4))}</h3>`);
      continue;
    }
    if (/^#### /.test(line)) {
      closeList();
      output.push(`<h4>${inlineFormat(line.slice(5))}</h4>`);
      continue;
    }

    // ── Horizontal rule ──
    if (/^---+$/.test(line.trim())) {
      closeList();
      output.push("<hr>");
      continue;
    }

    // ── Blockquote ──
    if (/^> /.test(line)) {
      closeList();
      output.push(`<blockquote>${inlineFormat(line.slice(2))}</blockquote>`);
      continue;
    }

    // ── Unordered list ──
    if (/^[\-\*] /.test(line)) {
      if (!inUl) { output.push("<ul>"); inUl = true; }
      output.push(`<li>${inlineFormat(line.slice(2))}</li>`);
      continue;
    }

    // ── Ordered list ──
    const olMatch = line.match(/^(\d+)\. (.*)/);
    if (olMatch) {
      if (!inOl) { output.push("<ol>"); inOl = true; }
      output.push(`<li>${inlineFormat(olMatch[2])}</li>`);
      continue;
    }

    // ── Regular paragraph ──
    closeList();
    output.push(`<p>${inlineFormat(line)}</p>`);
  }

  closeList();
  return output.join("\n");
}


// ── Utility ───────────────────────────────────────────────────────────────

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function printRecipe() {
  window.print();
}
