# 🍛 Sri Lankan Flavor Assistant

A Streamlit app that gives you a Sri Lankan recipe adjusted to your spice or sweetness level, plus nutrition info, using RAG so answers come from a real recipe corpus instead of the LLM just making things up.

**Live demo:** https://sl-flavor-agent-bydil.streamlit.app/

Built for IT41043 — Intelligent Systems (Agentic AI), Horizon Campus.

## What it does

Pick a dish (or use today's featured one), set the spice/sweetness slider, and click either:
- **Get Recipe** — full recipe adjusted to your slider level
- **Get Nutrition Info** — calories, protein, fat, carbs, sodium

## How it works

```
User picks dish + slider + button
        │
        ▼
message = {intent, dish, adjustment_level, is_sweet}
        │
        ▼
   router_agent()
   ├── intent == "recipe"    → recipe_agent()    → OpenRouter LLM
   └── intent == "nutrition" → nutrition_agent()  → Groq LLM
        │
        ▼
   both agents first do vectorstore.similarity_search()
   to pull relevant recipe context before generating
```

One router function reads the intent and calls the right agent. Both agents retrieve context from the vector store first (RAG), then pass it to an LLM to generate the actual answer.

## RAG pipeline

- 22 dish `.txt` files (20 mains + 2 desserts) make up the corpus
- Embedded with `sentence-transformers/all-MiniLM-L6-v2`
- Stored in a local Chroma vector store (`./chroma_db`), cached with `@st.cache_resource`
- On each request: similarity search → top-k chunks → inserted into the prompt as context → sent to the LLM

### Retrieval test

Ran 5 test queries directly against the vector store (`k=2`) to check retrieval quality:

| Query | Top result(s) | Relevant? |
|---|---|---|
| spicy chicken curry | chicken_curry.txt | ✅ Yes |
| vegetarian dish | fish_curry.txt, kottu_roti.txt (chicken) | ❌ No — corpus has no dietary tags |
| coconut sambol | pol_sambol.txt, milk_rice.txt | ⚠️ Partial |
| dessert | watalappan.txt, curd_and_treacle.txt | ✅ Yes |
| mild curry | potato_curry.txt, pumpkin_curry.txt | ✅ Yes |

3 out of 5 were clean hits. The one clear failure ("vegetarian dish") happened because nothing in the corpus is tagged by diet, the retriever just matches on generic curry text. Adding a `diet` metadata field would fix this.

## Models used

| Agent | Provider | Model | Why |
|---|---|---|---|
| Recipe | OpenRouter | `meta-llama/llama-3.1-8b-instruct` | Bigger task — writing a full recipe, needs more general generation |
| Nutrition | Groq | `llama-3.1-8b-instant` | Small task — just extracting numbers, so speed/cost matter more than reasoning |

## Setup

```bash
git clone https://github.com/dilekha2001/sl-flavor-agent.git
cd sl-flavor-agent
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

Run:
```bash
streamlit run app.py
```

## Limitations

- Small corpus (22 dishes) — anything outside that gets the closest guess, not a real match
- No memory between requests — each click is independent
- No fact-checking on generated text; nutrition numbers are sometimes approximations
- Needs both Groq and OpenRouter working — if either is down, that feature breaks
- Retrieval struggles with dietary/constraint queries (see test table above)
- No rate limiting on the public demo
- English only
