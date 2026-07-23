import os
import random
import datetime
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
# Custom styling — warm, light, food-inspired palette
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #FFF8F0;
    }
    h1 {
        color: #C1440E;
        font-weight: 800;
    }
    h2, h3 {
        color: #7A3E1D;
    }
    div.stButton > button {
        background-color: #FFFFFF;
        color: #7A3E1D;
        border: 1.5px solid #E8B08A;
        border-radius: 12px;
        padding: 0.5em 1em;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #E8734A;
        color: white;
        border: 1.5px solid #E8734A;
    }
    .dish-card {
        background-color: #FDEEE0;
        border-radius: 16px;
        padding: 1.2em 1.5em;
        margin-bottom: 1em;
        border: 1px solid #F0D5B8;
    }
    .signature-banner {
        background: linear-gradient(90deg, #FDEEE0, #FFF3E4);
        border-radius: 16px;
        padding: 1.3em 1.5em;
        border: 1px solid #F0D5B8;
        margin-bottom: 1em;
    }
    .result-box {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 1.3em 1.5em;
        border: 1px solid #F0E0CC;
        margin-top: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍛 Sri Lankan Flavor Assistant")
st.caption("Discover dishes, adjust the spice, and check the nutrition — all in one place.")

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
# Agent functions
# ---------------------------------------------------------
def router_agent(action_type, dish, spice_level=None):
    # decides which agent should handle this, and builds a structured message
    ...

def recipe_agent(dish_name, spice_level, vectorstore, llm_client, model_name):
    results = vectorstore.similarity_search(dish_name, k=3)
    context_text = "\n\n".join([r.page_content for r in results])

    prompt = f"""You are a Sri Lankan cuisine recipe assistant.

Relevant recipe context:
{context_text}

Dish requested: {dish_name}
Requested spice level (1=mild, 5=very spicy): {spice_level}

Provide a recipe using the context above, adjusting chili/spice quantities
realistically to match the requested spice level. Include ingredient list
with quantities and clear steps."""

    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, something went wrong generating your recipe: {e}"


def nutrition_agent(dish_name, vectorstore, llm_client, model_name):
    results = vectorstore.similarity_search(dish_name, k=2)
    context_text = "\n\n".join([r.page_content for r in results])

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
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, couldn't fetch nutrition info: {e}"


def get_signature_dish(available_dishes):
    day_index = datetime.date.today().toordinal() % len(available_dishes)
    return available_dishes[day_index]


# ---------------------------------------------------------
# Dish list (mirrors your data/ corpus)
# ---------------------------------------------------------
available_dishes = [
    "Chicken Curry", "Fish Curry", "Dhal Curry", "Egg Curry",
    "Beetroot Curry", "Potato Curry", "Cashew Curry", "Jackfruit Curry",
    "Pumpkin Curry", "Pol Sambol", "Seeni Sambol", "Gotukola Mallung",
    "Brinjal Moju", "Kottu Roti", "Hoppers", "String Hoppers",
    "Deviled Chicken", "Fish Cutlets", "Watalappan", "Milk Rice",
    "Curd and Treacle", "Lamprais"
]

if "selected_dish" not in st.session_state:
    st.session_state.selected_dish = None

# ---------------------------------------------------------
# Section 1: Today's Signature Dish
# ---------------------------------------------------------
signature_dish = get_signature_dish(available_dishes)

st.markdown(f"""
<div class="signature-banner">
    <h3>⭐ Today's Signature Dish</h3>
    <p style="font-size:1.1em;">Today's featured dish is <b>{signature_dish}</b></p>
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

cols = st.columns(3)
for i, dish in enumerate(available_dishes):
    if cols[i % 3].button(dish, key=f"dish_{i}"):
        st.session_state.selected_dish = dish

st.write("")
st.divider()

# ---------------------------------------------------------
# Section 3: Once a dish is selected, show controls + results
# ---------------------------------------------------------
if st.session_state.selected_dish:
    dish = st.session_state.selected_dish

    st.markdown(f"""
    <div class="dish-card">
        <h3>🍽️ {dish}</h3>
    </div>
    """, unsafe_allow_html=True)

    spice_level = st.slider("Adjust spice level 🌶️", 1, 5, 3, key="spice_slider")

    col1, col2 = st.columns(2)

    with col1:
        get_recipe_clicked = st.button("Get Recipe", key="get_recipe_btn", use_container_width=True)
    with col2:
        get_nutrition_clicked = st.button("Get Nutrition Info", key="get_nutrition_btn", use_container_width=True)

    if get_recipe_clicked:
        with st.spinner(f"Preparing your {dish} recipe..."):
            recipe_result = recipe_agent(
                dish, spice_level, vectorstore, openrouter_client, RECIPE_MODEL
            )
            st.markdown(f'<div class="result-box">{recipe_result}</div>', unsafe_allow_html=True)

    if get_nutrition_clicked:
        with st.spinner(f"Calculating nutrition for {dish}..."):
            nutrition_result = nutrition_agent(
                dish, vectorstore, groq_client, NUTRITION_MODEL
            )
            st.markdown(f'<div class="result-box">{nutrition_result}</div>', unsafe_allow_html=True)
else:
    st.info("👆 Select a dish above (either today's signature dish or from the browse list) to get started.")