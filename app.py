import os
import html
import base64
from pathlib import Path
import difflib
import datetime
from typing import Optional, Dict, Any
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

st.set_page_config(
    page_title="SL Flavor Agent — A taste of Sri Lanka",
    page_icon="❋",
    layout="wide",
)

# ---------------------------------------------------------
# Styling (ported from the SL Flavor Agent HTML design)
# ---------------------------------------------------------
st.markdown("""
<style>
:root {
    --forest: #183f32;
    --forest-2: #245744;
    --leaf: #dce9d9;
    --gold: #d5a33c;
    --terracotta: #b95e3d;
    --cream: #fbf8f1;
    --paper: #fffdf8;
    --ink: #202820;
    --muted: #73786e;
    --line: #e9e4d8;
    --shadow: 0 18px 50px rgba(32, 48, 35, .08);
}

/* ---- Streamlit chrome ---- */
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }
.stApp {
    background: var(--cream);
    color: var(--ink);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.block-container { max-width: 1180px; padding: 0 20px 0 !important; }
[data-testid="InputInstructions"] { display: none; }

/* ---- Header ---- */
.site-header {
    display: flex; align-items: center; justify-content: space-between; gap: 24px;
    height: 78px; border-bottom: 1px solid rgba(24, 63, 50, .08);
}
.brand { display: flex; align-items: center; gap: 12px; text-decoration: none !important; }
.brand-mark {
    width: 42px; height: 42px; border-radius: 14px; display: grid; place-items: center;
    background: var(--forest); color: #fff; font-size: 22px;
    box-shadow: 0 8px 18px rgba(24, 63, 50, .17);
}
.brand-name { font-weight: 800; letter-spacing: -.5px; color: var(--forest); font-size: 18px; }
.brand-tagline { color: var(--muted); font-size: 11px; margin-top: 3px; }
.site-header nav { display: flex; align-items: center; gap: 30px; }
.site-header nav a { text-decoration: none !important; color: #4e5a50; font-size: 13px; font-weight: 600; }
.site-header nav a:hover { color: var(--terracotta); }
.profile {
    width: 38px; height: 38px; border-radius: 50%; display: grid; place-items: center;
    background: #f0e4cc; color: var(--forest); font-weight: 800; font-size: 13px;
}

/* ---- Hero ---- */
.eyebrow {
    display: inline-flex; align-items: center; gap: 8px; color: var(--terracotta);
    text-transform: uppercase; letter-spacing: 1.8px; font-size: 10px; font-weight: 800;
    margin: 40px 0 17px;
}
.eyebrow:before { content: ""; width: 24px; height: 1px; background: var(--terracotta); }
.hero-title {
    color: var(--forest); font-size: clamp(38px, 4.4vw, 60px); font-weight: 800;
    letter-spacing: -2.6px; line-height: 1.06; max-width: 570px;
}
.hero-title em { color: var(--terracotta); font-family: Georgia, serif; font-weight: 500; }
.hero-copy { color: #6b7168 !important; font-size: 15px; line-height: 1.8; max-width: 480px; margin: 21px 0 26px; }

/* search card (a bordered container tagged with .search-marker) */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.search-marker) {
    background: var(--paper); border: 1px solid var(--line) !important; border-radius: 18px;
    padding: 6px 10px; box-shadow: 0 12px 35px rgba(43, 56, 40, .06); max-width: 560px;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.search-marker) div[data-baseweb="input"],
div[data-testid="stVerticalBlockBorderWrapper"]:has(.search-marker) div[data-baseweb="base-input"],
div[data-testid="stVerticalBlockBorderWrapper"]:has(.search-marker) div[data-baseweb="input"] > div,
div[data-testid="stVerticalBlockBorderWrapper"]:has(.search-marker) [data-testid="stTextInputRootElement"],
div[data-testid="stVerticalBlockBorderWrapper"]:has(.search-marker) [data-testid="stTextInput"] > div {
    background: transparent !important; background-color: transparent !important; border: 0 !important; box-shadow: none !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.search-marker) input {
    background: transparent !important; color: var(--ink) !important; font-size: 13px;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.search-marker) button,
div[data-testid="stVerticalBlockBorderWrapper"]:has(.search-marker) div.stButton > button {
    background: var(--forest) !important; border: 0 !important; border-radius: 12px !important;
    color: #fff !important; font-weight: 700; padding: .65rem 1rem; transition: transform .2s, background .2s;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.search-marker) button:hover,
div[data-testid="stVerticalBlockBorderWrapper"]:has(.search-marker) div.stButton > button:hover {
    background: var(--forest-2) !important; color: #fff !important; transform: translateY(-1px);
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.search-marker) button p { color: #fff !important; }
.search-marker, .personalize-marker, .result-marker { display: none; }

/* chips = secondary buttons, active chip = primary button */
div.stButton > button {
    border: 1px solid #e5dfd2; background: rgba(255, 253, 248, .6); color: #536056;
    border-radius: 999px; padding: .45rem .9rem; font-size: 12px; font-weight: 650;
    transition: all .15s ease;
}
div.stButton > button:hover { border-color: #b6c9b6; background: var(--leaf); color: var(--forest); }
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="stBaseButton-primary"] {
    border-color: #b6c9b6; background: var(--leaf); color: var(--forest);
}
div.stButton > button p { color: inherit; font-size: inherit; font-weight: inherit; }

/* feature card */
.food-feature {
    position: relative; height: 410px; border-radius: 26px; overflow: hidden;
    background: #d7b77b; box-shadow: var(--shadow); margin-top: 36px;
}
.food-feature img { width: 100%; height: 410px; display: block; object-fit: cover; }
.food-feature:after {
    content: ""; position: absolute; inset: 40% 0 0;
    background: linear-gradient(transparent, rgba(16, 34, 23, .82)); pointer-events: none;
}
.feature-top {
    position: absolute; z-index: 1; top: 20px; left: 20px; background: rgba(255, 253, 248, .94);
    color: var(--forest); border-radius: 999px; padding: 9px 12px; font-size: 10px;
    font-weight: 800; letter-spacing: .7px;
}
.feature-bottom { position: absolute; z-index: 1; bottom: 24px; left: 25px; right: 25px; color: #fff; }
.feature-bottom small { color: #f1dba8; font-size: 10px; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; }
.feature-title { font: 500 29px Georgia, serif; margin: 7px 0 6px; color: #fff; }
.feature-bottom p { margin: 0; color: rgba(255, 255, 255, .82) !important; font-size: 12px; line-height: 1.6; }

/* ---- Selected dish + results ---- */
.selected-card {
    display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
    background: var(--paper); border: 1px solid var(--line); border-radius: 18px; padding: 6px 22px 16px;
    margin-bottom: 14px;
}
.selected-card .eyebrow { margin: 14px 0 6px; }
.dish-title { font: 500 26px Georgia, serif; color: var(--forest); }
.level-pill {
    background: var(--leaf); color: var(--forest); border-radius: 999px; padding: 8px 14px;
    font-size: 11px; font-weight: 800; margin-top: 10px;
}
.hint {
    padding: 15px 17px; border-radius: 14px; background: #edf3e9; border: 1px solid #d7e3d3;
    color: var(--forest); font-size: 13px; line-height: 1.7; margin-bottom: 14px;
}
.agent-badge {
    display: inline-block; font-size: 11px; font-weight: 700; color: var(--forest);
    background: var(--leaf); border-radius: 999px; padding: 6px 12px; margin-bottom: 10px;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-marker) {
    background: var(--paper); border: 1px solid var(--line) !important; border-radius: 18px;
    padding: 10px 14px; box-shadow: 0 12px 35px rgba(43, 56, 40, .06);
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-marker) :is(p, li, td, th, strong, h1, h2, h3, h4) {
    color: var(--ink); line-height: 1.7;
}

/* ---- Categories ---- */
.section-heading h2, .section-title { color: var(--forest); font-size: 25px; font-weight: 800; letter-spacing: -.7px; margin: 0; }
.section-sub { margin: 7px 0 20px; color: var(--muted) !important; font-size: 12px; }
.category {
    border: 1px solid var(--line); border-radius: 18px; overflow: hidden; background: var(--paper);
    margin-bottom: 8px; transition: transform .2s, box-shadow .2s, border-color .2s;
}
.category:hover { transform: translateY(-4px); box-shadow: var(--shadow); border-color: #c8d5c5; }
.category img { display: block; width: 100%; height: 142px; object-fit: cover; background: #e7ddc7; }
.category-content { padding: 15px 16px 17px; }
.category-content strong { display: block; color: var(--forest); font-size: 13px; }
.category-content span { display: block; color: var(--muted); font-size: 11px; margin-top: 5px; }
.picker-title { color: var(--forest); font-weight: 800; font-size: 14px; margin: 22px 0 8px; }

/* ---- Personalize ---- */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.personalize-marker) {
    background: #e9efe5 !important; border: 1px solid #dce6d7 !important; border-radius: 22px; padding: 14px 20px;
    margin-top: 40px; margin-bottom: 40px;
}
.pz-title { margin: 0; color: var(--forest); font-size: 19px; font-weight: 800; letter-spacing: -.4px; }
.pz-copy { margin: 7px 0 0; color: #677367 !important; font-size: 12px; line-height: 1.6; }
.spice-labels { display: flex; justify-content: space-between; color: #536256; font-size: 10px; font-weight: 700; margin-top: 6px; }
div[data-testid="stSlider"] div[role="slider"] { background: var(--forest) !important; }

/* ---- Footer ---- */
.site-footer { border-top: 1px solid var(--line); padding: 23px 20px; text-align: center; color: #888b81; font-size: 11px; }
.site-footer strong { color: var(--forest); }

@media (max-width: 820px) {
    .site-header nav a { display: none; }
    .food-feature, .food-feature img { height: 350px; }
}
@media (max-width: 520px) {
    .hero-title { font-size: 42px; letter-spacing: -1.8px; }
    .food-feature, .food-feature img { height: 330px; }
    .category img { height: 115px; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Setup: API keys, clients, vectorstore
# ---------------------------------------------------------
groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
openrouter_key = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", None)

groq_client = Groq(api_key=groq_key)
openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)

RECIPE_MODEL = "meta-llama/llama-3.1-8b-instruct"   # OpenRouter - deep reasoning & reflection
NUTRITION_MODEL = "openai/gpt-oss-20b"           # Groq - fast, low-latency extraction


@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


vectorstore = load_vectorstore()

# ---------------------------------------------------------
# Agent Communication Protocol
# ---------------------------------------------------------
class AgentMessage:
    """Structured message passed between agents.

    sender / recipient identify which agent produced / should consume the
    message, and payload carries whatever structured data that agent needs.
    Every inter-agent exchange in this app (recipe_agent <-> reflection_agent,
    recipe_agent -> nutrition_agent -> recipe_agent) is expressed as one of
    these rather than a plain function return value, so the hand-off is
    explicit and inspectable (see README sequence diagram).
    """

    def __init__(self, sender: str, recipient: str, payload: Dict[str, Any]):
        self.sender = sender
        self.recipient = recipient
        self.payload = payload
        self.timestamp = datetime.datetime.now().isoformat()

    def __repr__(self):
        return f"AgentMessage({self.sender} -> {self.recipient}, keys={list(self.payload.keys())})"


# ---------------------------------------------------------
# Agent Functions
# ---------------------------------------------------------
def reflection_agent(message: AgentMessage, llm_client, model_name: str) -> AgentMessage:
    """Self-Correction Agent.

    Receives an AgentMessage from recipe_agent containing the generated
    recipe text and the retrieval context it was based on. Critiques the
    recipe against that context and replies with an AgentMessage containing
    the verified (possibly corrected) recipe.
    """
    recipe_text = message.payload["recipe_text"]
    context_text = message.payload["context_text"]

    prompt = f"""You are a senior Sri Lankan culinary reviewer.
Review the following generated recipe against the authentic reference context provided.

Reference Context:
{context_text}

Generated Recipe:
{recipe_text}

Task:
1. Check if any ingredient quantities or steps contradict traditional Sri Lankan cooking or the context.
2. If the recipe is accurate and well-structured, return it exactly as-is.
3. If there are hallucinations or missing key steps, output a corrected version and append a brief line at the end: "**Chef's Note (Self-Corrected):** [brief explanation of fix]".

Output ONLY the final verified recipe text."""

    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        verified_text = response.choices[0].message.content
        corrected = "Self-Corrected" in verified_text
    except Exception:
        verified_text = recipe_text
        corrected = False

    return AgentMessage(
        sender="reflection_agent",
        recipient="recipe_agent",
        payload={"verified_recipe": verified_text, "corrected": corrected}
    )


def recipe_agent(dish_name, adjustment_level, vectorstore, llm_client, model_name, is_sweet=False):
    """Generates a recipe via RAG, then hands it off to reflection_agent
    for verification before returning the final result."""
    results = vectorstore.similarity_search(dish_name, k=3)
    context_text = "\n\n".join([r.page_content for r in results])

    if is_sweet:
        adjustment_instruction = f"""Requested sweetness level (1=lightly sweet, 5=very sweet): {adjustment_level}
Provide a recipe using the context above, adjusting sugar/jaggery/treacle quantities realistically to match the requested sweetness level. Include ingredient list with quantities and clear steps."""
    else:
        adjustment_instruction = f"""Requested spice level (1=mild, 5=very spicy): {adjustment_level}
Provide a recipe using the context above, adjusting chili/spice quantities realistically to match the requested spice level. Include ingredient list with quantities and clear steps."""

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
        initial_recipe = response.choices[0].message.content

        # Agent-to-agent message: recipe_agent -> reflection_agent
        reflection_request = AgentMessage(
            sender="recipe_agent",
            recipient="reflection_agent",
            payload={"recipe_text": initial_recipe, "context_text": context_text}
        )
        reflection_reply = reflection_agent(reflection_request, llm_client, model_name)

        return {
            "recipe_text": reflection_reply.payload["verified_recipe"],
            "self_corrected": reflection_reply.payload["corrected"],
        }
    except Exception as e:
        return {
            "recipe_text": f"Sorry, something went wrong generating your recipe: {e}",
            "self_corrected": False,
        }


def nutrition_agent(dish_name, vectorstore, llm_client, model_name,
                     incoming_message: Optional[AgentMessage] = None) -> AgentMessage:
    """Estimates nutrition info via RAG. If it received an AgentMessage from
    recipe_agent carrying the exact generated recipe, it grounds the
    estimate in those specific ingredients/quantities instead of generic
    corpus context, and replies with an AgentMessage back to recipe_agent."""
    results = vectorstore.similarity_search(dish_name, k=2)
    context_text = "\n\n".join([r.page_content for r in results])

    used_recipe_ingredients = incoming_message is not None and bool(
        incoming_message.payload.get("ingredients")
    )

    if used_recipe_ingredients:
        ingredients = incoming_message.payload["ingredients"]
        prompt = f"""You are a concise nutrition assistant.

Estimate nutrition information (calories, protein, fat, carbohydrates, sodium) for the dish: {dish_name}

Base the estimate on these exact ingredients and quantities used in this specific generated recipe:
{ingredients}

Reference context:
{context_text}

STRICT OUTPUT RULES:
1. Do NOT include conversational intros.
2. Provide a brief breakdown per main ingredient.
3. End with a clear **Total Summary** table for the full dish."""
    else:
        prompt = f"""Extract and summarize only the nutrition information (calories, protein, fat, carbohydrates, sodium) for the dish: {dish_name}

Context:
{context_text}

If exact figures aren't present, give the closest approximate values found in the context.

STRICT OUTPUT RULES:
1. Do NOT include conversational intros.
2. Provide a clean, concise markdown table or summary."""

    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = response.choices[0].message.content
    except Exception as e:
        result_text = f"Sorry, couldn't fetch nutrition info: {e}"

    # Agent-to-agent message: nutrition_agent -> recipe_agent
    return AgentMessage(
        sender="nutrition_agent",
        recipient="recipe_agent",
        payload={
            "nutrition_summary": result_text,
            "used_recipe_ingredients": used_recipe_ingredients,
        }
    )


def router_agent(message, vectorstore, groq_client, openrouter_client):
    """Reads intent and dispatches to the right agent. Unwraps whatever
    structured result/AgentMessage that agent returns into a plain dict
    the UI layer can render."""
    intent = message["intent"]
    dish = message["dish"]

    if intent == "recipe":
        adjustment_level = message["adjustment_level"]
        is_sweet = message["is_sweet"]
        result = recipe_agent(
            dish, adjustment_level, vectorstore, openrouter_client, RECIPE_MODEL, is_sweet=is_sweet
        )
        return {"type": "recipe", **result}

    elif intent == "nutrition":
        incoming_msg = message.get("incoming_message", None)
        reply = nutrition_agent(
            dish, vectorstore, groq_client, NUTRITION_MODEL, incoming_message=incoming_msg
        )
        return {
            "type": "nutrition",
            "nutrition_summary": reply.payload["nutrition_summary"],
            "used_recipe_ingredients": reply.payload["used_recipe_ingredients"],
        }

    else:
        return {"type": "error", "message": "Sorry, I didn't understand that request."}


def get_signature_dish(available_dishes):
    day_index = datetime.date.today().toordinal() % len(available_dishes)
    return available_dishes[day_index]


# ---------------------------------------------------------
# Dish catalogue (grouped into the 4 categories from the design)
# ---------------------------------------------------------
CATEGORIES = [
    {
        "name": "Rice & Curry",
        "tagline": "Everyday island comfort",
        "img": "https://images.unsplash.com/photo-1546833999-b9f581a1996d?auto=format&fit=crop&w=700&q=80",
        "dishes": ["Chicken Curry", "Fish Curry", "Egg Curry", "Cashew Curry",
                   "Pol Sambol", "Seeni Sambol", "Lamprais", "Milk Rice"],
    },
    {
        "name": "Street Food",
        "tagline": "Big flavor, lively spirit",
        "img": "https://images.unsplash.com/photo-1601050690597-df0568f70950?auto=format&fit=crop&w=700&q=80",
        "dishes": ["Kottu Roti", "Hoppers", "String Hoppers", "Deviled Chicken", "Fish Cutlets"],
    },
    {
        "name": "Plant-based",
        "tagline": "Fresh, colorful and nourishing",
        "img": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?auto=format&fit=crop&w=700&q=80",
        "dishes": ["Dhal Curry", "Beetroot Curry", "Potato Curry", "Jackfruit Curry",
                   "Pumpkin Curry", "Gotukola Mallung", "Brinjal Moju"],
    },
    {
        "name": "Sweet Treats",
        "tagline": "A little sweetness to finish",
        "img": "https://images.unsplash.com/photo-1488477181946-6428a0291777?auto=format&fit=crop&w=700&q=80",
        "dishes": ["Watalappan", "Curd and Treacle"],
    },
]

DESSERTS = ["Watalappan", "Curd and Treacle"]
SWEET_DISHES = set(DESSERTS)
available_dishes = [d for c in CATEGORIES for d in c["dishes"]]
DISH_TO_CATEGORY = {d: c["name"] for c in CATEGORIES for d in c["dishes"]}

SPICE_NAMES = {1: "Very mild", 2: "Mild", 3: "Balanced", 4: "Spicy", 5: "Extra hot"}
SWEET_NAMES = {1: "Lightly sweet", 2: "Mildly sweet", 3: "Balanced", 4: "Sweet", 5: "Very sweet"}

# ---------------------------------------------------------
# Dish photos (local files in ./images, next to this app.py)
# ---------------------------------------------------------
APP_DIR = Path(__file__).parent
# Use ./images if it is a real folder, otherwise look next to app.py
IMAGE_DIR = APP_DIR / "images" if (APP_DIR / "images").is_dir() else APP_DIR
FALLBACK_IMAGE = "https://images.unsplash.com/photo-1603894584373-5ac82b2ae398?auto=format&fit=crop&w=1200&q=85"
DISH_IMAGES = {
    "Beetroot Curry": "beetroot-curry.jpg",
    "Brinjal Moju": "brinjal-moju.jpg",
    "Cashew Curry": "cashew-curry.jpg",
    "Chicken Curry": "chicken-curry.jpg",
    "Curd and Treacle": "curd-and-treacle.jpg",
    "Dhal Curry": "dhal-curry.jpg",
    "Egg Curry": "egg-curry.jpg",
    "Fish Curry": "fish-curry.jpg",
    "Fish Cutlets": "fish-cutlets.jpg",
    "Deviled Chicken": "deviled-chicken.jpg",
    "Gotukola Mallung": "gotukola-mallung.jpg",
    "Hoppers": "hoppers.jpg",
    "Jackfruit Curry": "jackfruit-curry.jpg",
    "Kottu Roti": "kottu-roti.jpg",
    "Lamprais": "lamprais.jpg",
    "Milk Rice": "milk-rice.jpg",
    "Pol Sambol": "pol-sambol.jpg",
    "Potato Curry": "potato-curry.jpg",
    "Pumpkin Curry": "pumpkin-curry.jpg",
    "Seeni Sambol": "seeni-sambol.jpg",
    "String Hoppers": "string-hoppers.jpg",
    "Watalappan": "watalappan.jpg",
}


@st.cache_data
def dish_image_src(dish: str) -> str:
    """Returns an <img> src for the dish: the local photo (embedded as a
    data URI) if we have one, otherwise the generic curry photo."""
    filename = DISH_IMAGES.get(dish)
    path = IMAGE_DIR / filename if filename else None
    if path and path.exists():
        encoded = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/jpeg;base64,{encoded}"
    return FALLBACK_IMAGE


# ---------------------------------------------------------
# Session state + callbacks
# ---------------------------------------------------------
defaults = {
    "selected_dish": None,
    "last_generated_recipe": {},
    "category": CATEGORIES[0]["name"],
    "active_chip": "recipe",
    "pending_action": None,
    "notice": None,
    "last_result": None,
    "dish_query": "",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def match_dish(query: str) -> Optional[str]:
    q = query.strip().lower()
    for d in available_dishes:
        if d.lower() == q:
            return d
    for d in available_dishes:
        if q in d.lower() or d.lower() in q:
            return d
    lowered = {d.lower(): d for d in available_dishes}
    close = difflib.get_close_matches(q, list(lowered.keys()), n=1, cutoff=0.5)
    return lowered[close[0]] if close else None


def select_dish(dish: str):
    st.session_state.selected_dish = dish
    st.session_state.dish_query = dish
    st.session_state.category = DISH_TO_CATEGORY[dish]
    st.session_state.notice = None


def set_category(name: str):
    st.session_state.category = name


def set_chip(action: str):
    st.session_state.active_chip = action
    st.session_state.pending_action = action


def handle_search():
    query = st.session_state.get("dish_query", "").strip()
    if not query:
        st.session_state.notice = (
            "Type a Sri Lankan dish to get started — for example, "
            "<strong>chicken curry</strong>, <strong>hoppers</strong>, or <strong>watalappan</strong>."
        )
        return
    match = match_dish(query)
    if match:
        select_dish(match)
    else:
        st.session_state.notice = (
            f"I couldn't find “{html.escape(query)}” yet. Try one of the dishes in "
            "<strong>Explore the island on a plate</strong> below."
        )


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<div class="site-header">
<a class="brand" href="#discover" aria-label="SL Flavor Agent home">
<div class="brand-mark">❋</div>
<div>
<div class="brand-name">SL Flavor Agent</div>
<div class="brand-tagline">A taste of Sri Lanka, tailored to you</div>
</div>
</a>
<nav aria-label="Main navigation">
<a href="#discover">Discover</a>
<a href="#categories">Explore dishes</a>
<a href="#personalize">My preferences</a>
<div class="profile" title="Guest profile">SL</div>
</nav>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Hero: search + quick actions  |  today's signature dish
# ---------------------------------------------------------
signature_dish = get_signature_dish(available_dishes)

hero_left, hero_right = st.columns([1.02, 0.98], gap="large")

with hero_left:
    st.markdown("""
<div id="discover"></div>
<div class="eyebrow">Your personal Sri Lankan food guide</div>
<div class="hero-title">Your next Sri Lankan favorite <em>starts here.</em></div>
<p class="hero-copy">Explore beloved local dishes, discover recipes made for your taste, and get a clearer picture of what’s on your plate — all in one place.</p>
""", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<span class="search-marker"></span>', unsafe_allow_html=True)
        s_col1, s_col2 = st.columns([3, 1.4])
        s_col1.text_input(
            "Search a dish", key="dish_query", on_change=handle_search,
            placeholder="Search a dish, e.g. dhal curry...", label_visibility="collapsed",
        )
        s_col2.button("Find my dish ↗", key="find_btn", on_click=handle_search, use_container_width=True)

    st.write("")
    chip_cols = st.columns(3)
    chip_cols[0].button(
        "🍲 Get a recipe", key="chip_recipe", on_click=set_chip, args=("recipe",),
        type="primary" if st.session_state.active_chip == "recipe" else "secondary",
        use_container_width=True,
    )
    chip_cols[1].button(
        "✳ Nutrition insights", key="chip_nutrition", on_click=set_chip, args=("nutrition",),
        type="primary" if st.session_state.active_chip == "nutrition" else "secondary",
        use_container_width=True,
    )
    chip_cols[2].button(
        "🌶 Adjust spice", key="chip_spice", on_click=set_chip, args=("spice",),
        type="primary" if st.session_state.active_chip == "spice" else "secondary",
        use_container_width=True,
    )

with hero_right:
    st.markdown(f"""
<div class="food-feature">
<img src="{dish_image_src(signature_dish)}" alt="{html.escape(signature_dish)}" />
<div class="feature-top">✦ TODAY’S SIGNATURE DISH</div>
<div class="feature-bottom">
<small>A little island inspiration</small>
<div class="feature-title">{html.escape(signature_dish)}</div>
<p>Bold spices, rich flavor, and a tradition worth sharing.</p>
</div>
</div>
""", unsafe_allow_html=True)
    st.button(
        "Select today's dish ↗", key="signature_btn", on_click=select_dish,
        args=(signature_dish,), use_container_width=True,
    )

# Results are rendered here (top of page) but filled at the end of the script,
# once the spice / sweetness slider value is known.
result_slot = st.container()

# ---------------------------------------------------------
# Categories + dish picker
# ---------------------------------------------------------
st.markdown("""
<div id="categories"></div>
<div style="height:30px"></div>
<div class="section-title">Explore the island on a plate</div>
<p class="section-sub">Find something familiar, or fall in love with something new.</p>
""", unsafe_allow_html=True)

cat_cols = st.columns(4)
for col, cat in zip(cat_cols, CATEGORIES):
    with col:
        st.markdown(f"""
<div class="category">
<img src="{cat['img']}" alt="{html.escape(cat['name'])}" />
<div class="category-content"><strong>{html.escape(cat['name'])}</strong><span>{html.escape(cat['tagline'])}</span></div>
</div>
""", unsafe_allow_html=True)
        st.button(
            "Browse dishes", key=f"cat_{cat['name']}", on_click=set_category, args=(cat["name"],),
            type="primary" if st.session_state.category == cat["name"] else "secondary",
            use_container_width=True,
        )

active_cat = next(c for c in CATEGORIES if c["name"] == st.session_state.category)
st.markdown(f'<div class="picker-title">{html.escape(active_cat["name"])} · pick a dish</div>', unsafe_allow_html=True)
dish_cols = st.columns(4)
for i, d in enumerate(active_cat["dishes"]):
    dish_cols[i % 4].button(
        d, key=f"dish_{d}", on_click=select_dish, args=(d,),
        type="primary" if st.session_state.selected_dish == d else "secondary",
        use_container_width=True,
    )

# ---------------------------------------------------------
# Make it yours: spice / sweetness slider
# ---------------------------------------------------------
selected = st.session_state.selected_dish
is_sweet = selected in SWEET_DISHES if selected else False
level_key = "sweet_level" if is_sweet else "spice_level"
names = SWEET_NAMES if is_sweet else SPICE_NAMES
current_level = st.session_state.get(level_key, 3)

with st.container(border=True):
    st.markdown('<span class="personalize-marker"></span><div id="personalize"></div>', unsafe_allow_html=True)
    p_left, p_right = st.columns([1.1, 1], gap="large")
    with p_left:
        if is_sweet:
            copy = "Prefer it lightly sweet or extra indulgent? Choose your sweetness level and tailor the recipe to your taste."
        else:
            copy = "Love it mild or want a little more fire? Choose your comfort level and tailor the recipe to your taste."
        st.markdown(f'<div class="pz-title">Make it yours.</div><p class="pz-copy">{copy}</p>', unsafe_allow_html=True)
    with p_right:
        left_label, right_label = ("🍯 Lightly sweet", "🍮 Very sweet") if is_sweet else ("🌿 Mild", "🌶 Extra hot")
        st.markdown(
            f'<div class="spice-labels"><span>{left_label}</span>'
            f'<span>{names[current_level]} · {current_level}/5</span><span>{right_label}</span></div>',
            unsafe_allow_html=True,
        )
        adjustment_level = st.slider(
            "Choose your sweetness level" if is_sweet else "Choose your spice level",
            1, 5, 3, key=level_key, label_visibility="collapsed",
        )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div class="site-footer"><strong>SL Flavor Agent</strong> · Made with a love for Sri Lankan food. <span>Recipe and nutrition guidance is an estimate.</span></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Results (rendered into the slot under the hero, with Agent Messaging)
# ---------------------------------------------------------
pending = st.session_state.pending_action
st.session_state.pending_action = None
dish = st.session_state.selected_dish

with result_slot:
    st.write("")
    if dish:
        st.markdown(f"""
<div class="selected-card">
<div><div class="eyebrow">Selected dish</div><div class="dish-title">🍽️ {html.escape(dish)}</div></div>
<div class="level-pill">{names[adjustment_level]} · {adjustment_level}/5</div>
</div>
""", unsafe_allow_html=True)

    if st.session_state.notice:
        st.markdown(f'<div class="hint">{st.session_state.notice}</div>', unsafe_allow_html=True)

    if pending == "spice":
        st.markdown(
            '<div class="hint">Spice adjustment selected. Use the slider in <strong>Make it yours</strong> '
            'below to choose your preferred level, then pick <strong>Get a recipe</strong>.</div>',
            unsafe_allow_html=True,
        )
    elif pending in ("recipe", "nutrition") and not dish:
        st.markdown(
            '<div class="hint">Search for a dish above, or choose one from the list below, to get started.</div>',
            unsafe_allow_html=True,
        )
    elif pending == "recipe":
        message = {"intent": "recipe", "dish": dish, "adjustment_level": adjustment_level, "is_sweet": is_sweet}
        with st.spinner(f"Preparing and verifying your {dish} recipe..."):
            result = router_agent(message, vectorstore, groq_client, openrouter_client)
        # Save generated recipe in session state so nutrition_agent can
        # later be sent it via an AgentMessage
        st.session_state.last_generated_recipe[dish] = result["recipe_text"]
        st.session_state.last_result = {"dish": dish, **result}
    elif pending == "nutrition":
        with st.spinner(f"Calculating nutrition for {dish}..."):
            # If a recipe was previously generated for this dish, send it to
            # nutrition_agent via an AgentMessage
            incoming_msg = None
            if dish in st.session_state.last_generated_recipe:
                incoming_msg = AgentMessage(
                    sender="recipe_agent",
                    recipient="nutrition_agent",
                    payload={"ingredients": st.session_state.last_generated_recipe[dish]}
                )
            message = {"intent": "nutrition", "dish": dish, "incoming_message": incoming_msg}
            result = router_agent(message, vectorstore, groq_client, openrouter_client)
        st.session_state.last_result = {"dish": dish, **result}

    res = st.session_state.last_result
    if dish and res and res["dish"] == dish and res.get("type") in ("recipe", "nutrition"):
        with st.container(border=True):
            st.markdown('<span class="result-marker"></span>', unsafe_allow_html=True)
            if res["type"] == "recipe":
                badge = ("✅ verified by reflection_agent — corrections applied" if res["self_corrected"]
                         else "✅ verified by reflection_agent — no changes needed")
                st.markdown(f'<span class="agent-badge">{badge}</span>', unsafe_allow_html=True)
                st.markdown(res["recipe_text"])
            else:
                badge = ("📩 grounded in your generated recipe's exact ingredients" if res["used_recipe_ingredients"]
                         else "📖 grounded in corpus reference data")
                st.markdown(f'<span class="agent-badge">{badge}</span>', unsafe_allow_html=True)
                st.markdown(res["nutrition_summary"])
    elif not dish and not st.session_state.notice and not pending:
        st.markdown(
            '<div class="hint">👆 Search for a dish, use today’s signature dish, or browse the list below '
            'to get started.</div>',
            unsafe_allow_html=True,
        )
    elif dish and not pending and not (res and res["dish"] == dish):
        st.markdown(
            '<div class="hint">Dish selected. Choose <strong>Get a recipe</strong> or '
            '<strong>Nutrition insights</strong> to continue.</div>',
            unsafe_allow_html=True,
        )
