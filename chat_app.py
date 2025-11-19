"""
Streamlit Chatbox with:
 - OpenAI-powered replies
 - Integration with electric_vehicles_dataset.json (expected at /mnt/data/electric_vehicles_dataset.json)
 - Sidebar UI for filters / system prompt / model selection
 - Option to include dataset context in AI prompt
 - Save chat history to local JSON
"""

import os
import json
import time
from typing import List, Dict, Any, Optional

import streamlit as st

# Try to import OpenAI, but handle absence gracefully
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------------------------
# Constants & Defaults
# ---------------------------
DATA_PATH = "C:\\Users\\dell\\Desktop\\Project Database\\Electric vehicle\\electric_vehicles_dataset.csv"




HISTORY_PATH = "chat_history.json"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that answers user queries concisely. When provided with EV dataset context, use it to answer precisely."
DEFAULT_MODEL = "gpt-4o-mini"  # You can change this; or use gpt-4o / gpt-4o-mini depending on availability

# ---------------------------
# Helpers
# ---------------------------
import pandas as pd

@st.cache_data
def load_dataset(path: str):
    try:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")   # convert DataFrame → list of dicts
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return []

def search_dataset(data: List[Dict[str, Any]], query: str, max_results:int=10) -> List[Dict[str, Any]]:
    q = query.lower().strip()
    results = []
    for row in data:
        # search in Model, Manufacturer, Year, Battery_Type, Color
        combined = " ".join(
            str(row.get(k, "")).lower() for k in ("Model", "Manufacturer", "Year", "Battery_Type", "Color")
        )
        if q in combined:
            results.append(row)
        # numeric operations: allow queries like "range > 500" or "range >= 300"
        # simple parse: "range > 300" or "range < 200"
        if q.startswith("range") or "range" in q and any(op in q for op in [">", "<", ">=", "<=", "=="]):
            try:
                # naive: look for number
                import re
                num = int(re.search(r"(\d{2,4})", q).group(1))
                if ">" in q and row.get("Range_km") is not None and row.get("Range_km") > num:
                    results.append(row)
                if "<" in q and row.get("Range_km") is not None and row.get("Range_km") < num:
                    results.append(row)
            except Exception:
                pass

    # deduplicate and limit
    unique = []
    ids = set()
    for r in results:
        vid = r.get("Vehicle_ID") or json.dumps(r, sort_keys=True)
        if vid not in ids:
            unique.append(r)
            ids.add(vid)
        if len(unique) >= max_results:
            break
    return unique

def format_vehicle_short(v: Dict[str, Any]) -> str:
    return f"{v.get('Manufacturer','?')} {v.get('Model','?')} ({v.get('Year','?')}) — {v.get('Range_km','?')} km range, {v.get('Battery_Capacity_kWh','?')} kWh"

def append_history(entry: Dict[str, Any], path: str = HISTORY_PATH) -> None:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                hist = json.load(f)
        else:
            hist = []
        hist.append(entry)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")

def load_history(path: str = HISTORY_PATH) -> List[Dict[str, Any]]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

# ---------------------------
# OpenAI call wrapper
# ---------------------------
def call_openai_chat(model: str, messages: List[Dict[str,str]], temperature: float = 0.2, max_tokens: int = 512) -> str:
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI package not installed. Install with `pip install openai`.")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY".lower())
    if not api_key:
        raise RuntimeError("OpenAI API key not found. Set environment variable OPENAI_API_KEY.")
    openai.api_key = api_key

    # Use ChatCompletion for compatibility
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Get assistant content
        content = resp.choices[0].message.get("content", "")
        return content
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="EV Chatbox", layout="wide")

# Sidebar
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model (change if unavailable)", options=[DEFAULT_MODEL, "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"], index=0)
    system_prompt = st.text_area("System prompt (assistant persona)", value=DEFAULT_SYSTEM_PROMPT, height=120)
    include_data = st.checkbox("Include top-matching EV dataset entries in AI prompt", value=True)
    max_context = st.slider("Max dataset entries included in prompt", min_value=0, max_value=8, value=3)
    save_history_toggle = st.checkbox("Save chat history to JSON", value=True)
    st.markdown("---")
    st.write("Quick dataset search (sidebar):")
    quick_q = st.text_input("Search EV dataset (model/manufacturer/range...)")
    if st.button("Search EV dataset"):
        data = load_dataset(DATA_PATH)
        if not data:
            st.warning("Dataset not found or failed to load.")
        else:
            results = search_dataset(data, quick_q, max_results=20)
            st.write(f"Found {len(results)} result(s).")
            for r in results:
                st.write(format_vehicle_short(r))
            st.stop()

    st.markdown("---")
    st.write("OpenAI API key")
    st.caption("Set `OPENAI_API_KEY` as an environment variable on your system, or enter below for this session.")
    temp_api = st.text_input("OpenAI API key (optional for this session only)", type="password")
    if temp_api:
        os.environ["OPENAI_API_KEY"] = temp_api
        st.success("API key set for this session (temporary).")

    st.markdown("---")
    st.write("App controls")
    if st.button("Clear chat (session only)"):
        st.session_state.clear()
        st.experimental_rerun()

# Main app area
st.title("⚡ EV Chatbox — OpenAI + EV dataset")
st.markdown(
    """
Use the chatbox below to ask general questions or EV-specific queries.

**Pro tips**
- Start queries with `/ev ` to force a dataset search (e.g. `/ev Tesla Model Y`).
- Or ask the assistant normally (it will include dataset context when enabled).
"""
)

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str, "time": ts}

# Show previous messages (chat style)
for m in st.session_state.messages:
    role = m.get("role", "assistant")
    with st.chat_message(role):
        st.write(m.get("content", ""))

# Input
user_input = st.chat_input("Type a message...")

# Load dataset once
dataset = load_dataset(DATA_PATH)

if user_input:
    timestamp = time.time()
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input, "time": timestamp})
    with st.chat_message("user"):
        st.write(user_input)

    # Special: /ev commands -> dataset search + immediate answer (no API)
    if user_input.strip().lower().startswith("/ev"):
        query = user_input.strip()[3:].strip()
        if not query:
            bot_text = "Usage: `/ev <search terms>` — e.g. `/ev Tesla Model Y` or `/ev range > 400`"
        else:
            matches = search_dataset(dataset, query, max_results=10)
            if not matches:
                bot_text = f"No EVs matched your query: '{query}'. Try a different term."
            else:
                # Format results
                lines = [f"Found {len(matches)} result(s):"]
                for v in matches:
                    lines.append(f"- {format_vehicle_short(v)}")
                bot_text = "\n".join(lines)

        # Save/display assistant response
        st.session_state.messages.append({"role": "assistant", "content": bot_text, "time": time.time()})
        with st.chat_message("assistant"):
            st.write(bot_text)

        if save_history_toggle:
            append_history({"user": user_input, "assistant": bot_text, "time": time.time()})
    else:
        # Normal flow: prepare messages for OpenAI
        messages = []
        messages.append({"role": "system", "content": system_prompt})

        # Optionally add dataset context: find top matches and include them as "system" info
        if include_data and dataset:
            top_matches = search_dataset(dataset, user_input, max_results=max_context)
            if top_matches:
                ctx_lines = ["Top dataset matches (fields: Manufacturer, Model, Year, Range_km, Battery_Capacity_kWh):"]
                for v in top_matches:
                    ctx_lines.append(
                        f"{v.get('Manufacturer')}|{v.get('Model')}|{v.get('Year')}|range:{v.get('Range_km')}km|battery:{v.get('Battery_Capacity_kWh')}kWh"
                    )
                messages.append({"role": "system", "content": "\n".join(ctx_lines)})

        # Append conversation history (only roles and content) to provide context for the assistant
        # Keep last N messages small to avoid token bloat
        N_HISTORY = 12
        for h in st.session_state.messages[-N_HISTORY:]:
            messages.append({"role": h["role"], "content": h["content"]})

        # Add current user message
        messages.append({"role": "user", "content": user_input})

        # Call OpenAI
        bot_text = ""
        try:
            bot_text = call_openai_chat(model=model, messages=messages)
        except Exception as e:
            bot_text = f"Error calling OpenAI: {e}\n\n(You can still use `/ev` for dataset queries.)"

        # Save/display assistant response
        st.session_state.messages.append({"role": "assistant", "content": bot_text, "time": time.time()})
        with st.chat_message("assistant"):
            st.write(bot_text)

        # Save to JSON history if enabled
        if save_history_toggle:
            append_history({"user": user_input, "assistant": bot_text, "time": time.time()})
