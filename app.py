import os
import json
import datetime
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
import streamlit as st
from groq import Groq
from openai import OpenAI

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

st.set_page_config(page_title="Sri Lankan Flavor Assistant", page_icon="🍛", layout="centered")

# ---------------------------------------------------------
# Theme-adaptive, responsive styling
# Uses prefers-color-scheme so colors adjust automatically to
# the user's system/browser light or dark setting, and media
# queries so spacing/font sizes adapt for mobile screens.
# ---------------------------------------------------------
st.markdown("""
<style>
    :root {
        --accent: #D2691E;
        --accent-soft: #F2D9C4;
    }

    /* ---------- Light mode (default) ---------- */
    .card {
        background-color: rgba(210, 105, 30, 0.06);
        border: 1px solid rgba(210, 105, 30, 0.25);
        border-radius: 14px;
        padding: 1.4em 1.6em;
        margin-bottom: 1.2em;
    }
    .result-box {
        background-color: rgba(0, 0, 0, 0.02);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 12px;
        padding: 1.3em 1.6em;
        margin-top: 0.8em;
        line-height: 1.6;
    }
    div.stButton > button {
        border: 1px solid var(--accent-soft);
        border-radius: 10px;
        padding: 0.5em 1em;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    div.stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
    }
    h1 {
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .subtitle {
        opacity: 0.7;
        font-size: 0.95em;
        margin-top: -0.6em;
        margin-bottom: 1.2em;
    }

    /* ---------- Dark mode override ---------- */
    @media (prefers-color-scheme: dark) {
        .card {
            background-color: rgba(210, 105, 30, 0.10);
            border: 1px solid rgba(210, 105, 30, 0.35);
        }
        .result-box {
            background-color: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.10);
        }
    }

    /* ---------- Mobile responsiveness ---------- */
    @media (max-width: 640px) {
        h1 {
            font-size: 1.5em !important;
        }
        .card {
            padding: 1em 1.1em;
        }
        div.stButton > button {
            font-size: 0.85em;
            padding: 0.4em 0.6em;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("🍛 Sri Lankan Flavor Assistant")
st.markdown('<p class="subtitle">Discover dishes, adjust the spice, sweetness and check the nutrition — all in one place.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Setup: API keys, clients, vectorstore
# ---------------------------------------------------------
groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
openrouter_key = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", None)

groq_client = Groq(api_key=groq_key)
openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)

RECIPE_MODEL = "meta-llama/llama-3.1-8b-instruct"   # OpenRouter - deep reasoning / final recipe generation
NUTRITION_MODEL = "llama-3.1-8b-instant"             # Groq - fast, cheap extraction task

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

vectorstore = load_vectorstore()

# ---------------------------------------------------------
# Agent-to-agent messaging protocol
# ---------------------------------------------------------
@dataclass
class AgentMessage:
    """Structured message passed between agents.

    sender / receiver: which agent produced / should consume this message
    type: "request" or "response"
    payload: arbitrary structured data relevant to the message type
    """
    sender: str
    receiver: str
    type: str
    payload: dict


# ---------------------------------------------------------
# Agent functions
# ---------------------------------------------------------
def reflect_on_recipe(recipe_text, context_text, dish_name, llm_client, model_name):
    """Reflection pattern: the same agent re-reads its own output against the
    retrieved source context and either confirms it's accurate or produces a
    corrected version. This runs once, after generation, before the recipe is
    handed off to nutrition_agent or shown to the user.

    Returns (final_recipe_text, was_revised: bool, critique_note: str)
    """
    critique_prompt = f"""You are reviewing a generated recipe for factual consistency
against its source context. Be strict but fair.

Dish: {dish_name}

Source context (ground truth for this dish):
{context_text}

Generated recipe to review:
{recipe_text}

Check for: ingredients that contradict the context, missing key ingredients
mentioned in the context, or steps that don't make sense for this dish.

Respond in EXACTLY this format:
ISSUES_FOUND: yes or no
NOTE: one short sentence explaining what was wrong, or "None" if no issues
RECIPE: the corrected full recipe if issues were found, otherwise repeat the
original recipe unchanged"""

    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": critique_prompt}]
        )
        critique_output = response.choices[0].message.content
    except Exception:
        # If reflection itself fails, fall back to the original recipe
        # rather than blocking the whole request.
        return recipe_text, False, "Reflection step unavailable."

    issues_found = False
    note = "None"
    final_recipe = recipe_text

    try:
        lower = critique_output.lower()
        issues_found = "issues_found: yes" in lower

        if "note:" in lower and "recipe:" in lower:
            note_section = critique_output.split("NOTE:", 1)[1].split("RECIPE:", 1)[0].strip()
            recipe_section = critique_output.split("RECIPE:", 1)[1].strip()
            note = note_section if note_section else "None"
            if issues_found and recipe_section:
                final_recipe = recipe_section
    except Exception:
        # Parsing failed - keep the original recipe, don't revise blindly.
        issues_found = False
        note = "Could not parse reflection output."

    return final_recipe, issues_found, note


def recipe_agent(dish_name, adjustment_level, vectorstore, llm_client, model_name, is_sweet=False):
    """Generates a recipe, then extracts its ingredients into a structured
    AgentMessage addressed to nutrition_agent so nutrition can be computed
    from the ingredients actually used in *this* recipe (not a generic lookup)."""
    results = vectorstore.similarity_search(dish_name, k=3)
    context_text = "\n\n".join([r.page_content for r in results])

    if is_sweet:
        adjustment_instruction = f"""Requested sweetness level (1=lightly sweet, 5=very sweet): {adjustment_level}

Provide a recipe using the context above, adjusting sugar/jaggery/treacle quantities
realistically to match the requested sweetness level. Include ingredient list
with quantities and clear steps."""
    else:
        adjustment_instruction = f"""Requested spice level (1=mild, 5=very spicy): {adjustment_level}

Provide a recipe using the context above, adjusting chili/spice quantities
realistically to match the requested spice level. Include ingredient list
with quantities and clear steps."""

    prompt = f"""You are a Sri Lankan cuisine recipe assistant.

Relevant recipe context:
{context_text}

Dish requested: {dish_name}
{adjustment_instruction}"""

    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        recipe_text = response.choices[0].message.content
    except Exception as e:
        return f"Sorry, something went wrong generating your recipe: {e}", None

    # Reflection pattern: critique the generated recipe against the retrieved
    # context and self-correct before it's extracted/handed off/shown.
    recipe_text, was_revised, critique_note = reflect_on_recipe(
        recipe_text, context_text, dish_name, llm_client, model_name
    )
    if was_revised:
        recipe_text += f"\n\n*(Self-corrected by reflection step: {critique_note})*"

    # Ask the same model to extract a structured ingredient list from the
    # recipe it just wrote, so nutrition_agent can reason over real quantities
    # instead of doing its own generic RAG lookup on the dish name alone.
    extract_prompt = f"""From the recipe below, output ONLY a JSON array of ingredients
with their quantities, in this exact format and nothing else (no markdown, no explanation):
[{{"name": "ingredient name", "quantity": "amount", "unit": "unit"}}]

Recipe:
{recipe_text}"""

    ingredients = []
    try:
        extraction = llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": extract_prompt}]
        )
        raw = extraction.choices[0].message.content.strip()
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        ingredients = json.loads(raw)
    except Exception:
        # If extraction/parsing fails, nutrition_agent will fall back to its
        # own RAG-based lookup instead of failing the whole request.
        ingredients = []

    outgoing_message = AgentMessage(
        sender="recipe_agent",
        receiver="nutrition_agent",
        type="request",
        payload={"dish": dish_name, "ingredients": ingredients}
    )
    return recipe_text, outgoing_message


def nutrition_agent(dish_name, vectorstore, llm_client, model_name, incoming_message: Optional[AgentMessage] = None):
    """Computes nutrition info. If it receives a structured AgentMessage from
    recipe_agent containing real ingredient quantities, it uses those.
    Otherwise (standalone "Get Nutrition Info" click) it falls back to a
    RAG lookup on the dish name alone."""
    results = vectorstore.similarity_search(dish_name, k=2)
    context_text = "\n\n".join([r.page_content for r in results])

    if incoming_message is not None and incoming_message.payload.get("ingredients"):
        ingredients = incoming_message.payload["ingredients"]
        prompt = f"""Estimate nutrition information (calories, protein, fat, carbohydrates, sodium)
for the dish: {dish_name}

Base the estimate on these exact ingredients and quantities used in this specific recipe:
{ingredients}

Reference context (for typical values if a given ingredient is ambiguous):
{context_text}"""
    else:
        prompt = f"""Extract and summarize only the nutrition information
(calories, protein, fat, carbohydrates, sodium) for the dish: {dish_name}

Context:
{context_text}

If exact figures aren't present, give the closest approximate values found in the context."""

    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        nutrition_text = response.choices[0].message.content
    except Exception as e:
        return f"Sorry, couldn't fetch nutrition info: {e}"

    return nutrition_text


def router_agent(message, vectorstore, groq_client, openrouter_client):
    """Routes a UI request to the correct agent(s).

    For a "recipe" intent, this also triggers the agent-to-agent handoff:
    recipe_agent -> AgentMessage -> nutrition_agent, so nutrition is computed
    from the ingredients actually generated for this recipe.
    """
    intent = message["intent"]
    dish = message["dish"]

    if intent == "recipe":
        adjustment_level = message["adjustment_level"]
        is_sweet = message["is_sweet"]
        recipe_text, agent_message = recipe_agent(
            dish, adjustment_level, vectorstore, openrouter_client, RECIPE_MODEL, is_sweet=is_sweet
        )

        nutrition_text = None
        if agent_message is not None:
            nutrition_text = nutrition_agent(
                dish, vectorstore, groq_client, NUTRITION_MODEL, incoming_message=agent_message
            )

        return {"recipe": recipe_text, "nutrition": nutrition_text}

    elif intent == "nutrition":
        nutrition_text = nutrition_agent(dish, vectorstore, groq_client, NUTRITION_MODEL)
        return {"recipe": None, "nutrition": nutrition_text}

    else:
        return {"recipe": "Sorry, I didn't understand that request.", "nutrition": None}


def get_signature_dish(available_dishes):
    day_index = datetime.date.today().toordinal() % len(available_dishes)
    return available_dishes[day_index]


# ---------------------------------------------------------
# Dish list (mirrors your data/ corpus)
# ---------------------------------------------------------
MAIN_DISHES = [
    "Chicken Curry", "Fish Curry", "Dhal Curry", "Egg Curry",
    "Beetroot Curry", "Potato Curry", "Cashew Curry", "Jackfruit Curry",
    "Pumpkin Curry", "Pol Sambol", "Seeni Sambol", "Gotukola Mallung",
    "Brinjal Moju", "Kottu Roti", "Hoppers", "String Hoppers",
    "Deviled Chicken", "Fish Cutlets", "Lamprais", "Milk Rice"
]

DESSERTS = ["Watalappan", "Curd and Treacle"]

available_dishes = MAIN_DISHES + DESSERTS

# Dishes where "sweetness" is the relevant adjustment instead of "spice level"
SWEET_DISHES = set(DESSERTS)

if "selected_dish" not in st.session_state:
    st.session_state.selected_dish = None

# ---------------------------------------------------------
# Section 1: Today's Signature Dish
# ---------------------------------------------------------
signature_dish = get_signature_dish(available_dishes)

st.markdown(f"""
<div class="card">
    <h3 style="margin-top:0;">⭐ Today's Signature Dish</h3>
    <p style="font-size:1.05em; margin-bottom:0;">Today's featured dish is <b>{signature_dish}</b></p>
</div>
""", unsafe_allow_html=True)

if st.button("Select Today's Dish", key="signature_btn"):
    st.session_state.selected_dish = signature_dish

st.write("")

# ---------------------------------------------------------
# Section 2: Browse and choose a dish, split by category
# ---------------------------------------------------------
st.markdown("### 📖 Browse Available Dishes")
st.caption("Choose a dish to see its recipe and nutrition info:")

st.markdown("**🍛 Main Dishes**")
main_cols = st.columns(3)
for i, dish in enumerate(MAIN_DISHES):
    if main_cols[i % 3].button(dish, key=f"main_{i}", use_container_width=True):
        st.session_state.selected_dish = dish

st.write("")
st.markdown("**🍮 Desserts**")
dessert_cols = st.columns(3)
for i, dish in enumerate(DESSERTS):
    if dessert_cols[i % 3].button(dish, key=f"dessert_{i}", use_container_width=True):
        st.session_state.selected_dish = dish

st.write("")
st.divider()

# ---------------------------------------------------------
# Section 3: Once a dish is selected, show controls + results
# ---------------------------------------------------------
if st.session_state.selected_dish:
    dish = st.session_state.selected_dish

    st.markdown(f"""
    <div class="card">
        <h3 style="margin:0;">🍽️ {dish}</h3>
    </div>
    """, unsafe_allow_html=True)

    is_sweet = dish in SWEET_DISHES

    if is_sweet:
        adjustment_level = st.slider("Adjust sweetness level 🍯", 1, 5, 3, key="sweetness_slider")
    else:
        adjustment_level = st.slider("Adjust spice level 🌶️", 1, 5, 3, key="spice_slider")

    col1, col2 = st.columns(2)

    with col1:
        get_recipe_clicked = st.button("Get Recipe", key="get_recipe_btn", use_container_width=True)
    with col2:
        get_nutrition_clicked = st.button("Get Nutrition Info", key="get_nutrition_btn", use_container_width=True)

    if get_recipe_clicked:
        message = {"intent": "recipe", "dish": dish, "adjustment_level": adjustment_level, "is_sweet": is_sweet}
        with st.spinner(f"Preparing your {dish} recipe..."):
            result = router_agent(message, vectorstore, groq_client, openrouter_client)
            st.markdown(f'<div class="result-box">{result["recipe"]}</div>', unsafe_allow_html=True)

            if result["nutrition"]:
                st.markdown("**🧾 Nutrition (based on this recipe's actual ingredients)**")
                st.caption("Computed by nutrition_agent from a structured message sent by recipe_agent.")
                st.markdown(f'<div class="result-box">{result["nutrition"]}</div>', unsafe_allow_html=True)

    if get_nutrition_clicked:
        message = {"intent": "nutrition", "dish": dish}
        with st.spinner(f"Calculating nutrition for {dish}..."):
            result = router_agent(message, vectorstore, groq_client, openrouter_client)
            st.markdown(f'<div class="result-box">{result["nutrition"]}</div>', unsafe_allow_html=True)
else:
    st.info("👆 Select a dish above (either today's signature dish or from the browse list) to get started.")