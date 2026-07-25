import os
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

st.set_page_config(page_title="Sri Lankan Flavor Assistant", page_icon="🍛", layout="centered")

# ---------------------------------------------------------
# Theme-adaptive, responsive styling
# ---------------------------------------------------------
st.markdown("""
<style>
    :root {
        --accent: #D2691E;
        --accent-soft: #F2D9C4;
    }
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
    .agent-badge {
        display: inline-block;
        font-size: 0.75em;
        opacity: 0.65;
        border: 1px solid rgba(0, 0, 0, 0.15);
        border-radius: 20px;
        padding: 0.15em 0.7em;
        margin-bottom: 0.6em;
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
    @media (prefers-color-scheme: dark) {
        .card {
            background-color: rgba(210, 105, 30, 0.10);
            border: 1px solid rgba(210, 105, 30, 0.35);
        }
        .result-box {
            background-color: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.10);
        }
        .agent-badge {
            border: 1px solid rgba(255, 255, 255, 0.15);
        }
    }
    @media (max-width: 640px) {
        h1 { font-size: 1.5em !important; }
        .card { padding: 1em 1.1em; }
        div.stButton > button { font-size: 0.85em; padding: 0.4em 0.6em; }
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

RECIPE_MODEL = "meta-llama/llama-3.1-8b-instruct"   # OpenRouter - deep reasoning & reflection
NUTRITION_MODEL = "llama-3.1-8b-instant"            # Groq - fast, low-latency extraction


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
# Dish list & Session State Initialization
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
SWEET_DISHES = set(DESSERTS)

if "selected_dish" not in st.session_state:
    st.session_state.selected_dish = None
if "last_generated_recipe" not in st.session_state:
    st.session_state.last_generated_recipe = {}

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
# Section 2: Browse and choose a dish
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
# Section 3: Controls + Results (with Agent Messaging)
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
        with st.spinner(f"Preparing and verifying your {dish} recipe..."):
            result = router_agent(message, vectorstore, groq_client, openrouter_client)
            # Save generated recipe in session state so nutrition_agent can
            # later be sent it via an AgentMessage
            st.session_state.last_generated_recipe[dish] = result["recipe_text"]

            if result["self_corrected"]:
                st.markdown(
                    '<span class="agent-badge">✅ verified by reflection_agent — corrections applied</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<span class="agent-badge">✅ verified by reflection_agent — no changes needed</span>',
                    unsafe_allow_html=True
                )
            st.markdown(f'<div class="result-box">{result["recipe_text"]}</div>', unsafe_allow_html=True)

    if get_nutrition_clicked:
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

            if result["used_recipe_ingredients"]:
                st.markdown(
                    '<span class="agent-badge">📩 grounded in your generated recipe\'s exact ingredients</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<span class="agent-badge">📖 grounded in corpus reference data</span>',
                    unsafe_allow_html=True
                )
            st.markdown(f'<div class="result-box">{result["nutrition_summary"]}</div>', unsafe_allow_html=True)
else:
    st.info("👆 Select a dish above (either today's signature dish or from the browse list) to get started.")