import pickle
import string

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Next Word Predictor", page_icon="✍️", layout="centered")

# ── Cached loaders (model + tokenizer + max_len only load once) ────────────
@st.cache_resource
def load_artifacts():
    model = load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    index_word = {idx: word for word, idx in tokenizer.word_index.items()}
    return model, tokenizer, max_len, index_word


model, tokenizer, max_len, index_word = load_artifacts()
translator = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    return text.lower().translate(translator)


def predict_next_words(text: str, num_words: int = 1, top_k: int = 5):
    """
    Greedily extends `text` by `num_words` words using the model's top
    prediction at each step, and also returns the top_k candidates for
    the very next word.
    """
    seq = tokenizer.texts_to_sequences([clean_text(text)])[0]
    if not seq:
        return text, []

    # Top-k candidates for just the next word
    padded = pad_sequences([seq], maxlen=max_len, padding="pre")
    probs = model.predict(padded, verbose=0)[0]
    top_idx = np.argsort(probs)[-top_k:][::-1]
    top_candidates = [(index_word.get(i, "<unk>"), float(probs[i])) for i in top_idx]

    # Greedy multi-word continuation
    generated = text
    working_seq = list(seq)
    for _ in range(num_words):
        padded = pad_sequences([working_seq], maxlen=max_len, padding="pre")
        probs = model.predict(padded, verbose=0)[0]
        next_id = int(np.argmax(probs))
        next_word = index_word.get(next_id, "")
        if not next_word:
            break
        generated += " " + next_word
        working_seq.append(next_id)

    return generated, top_candidates


# ── UI ───────────────────────────────────────────────────────────────────
st.title("✍️ Next Word Predictor")
st.caption("LSTM model trained on a dataset of quotes — predicts what comes next.")

with st.sidebar:
    st.header("Settings")
    num_words = st.slider("Words to generate", min_value=1, max_value=20, value=5)
    top_k = st.slider("Number of suggestions to show", min_value=3, max_value=10, value=3)
    st.markdown("---")
    st.markdown(
        "**Model:** Embedding(50) → LSTM(128) → Dense(softmax)\n\n"
        f"**Vocab size:** {tokenizer.num_words}\n\n"
        f"**Max sequence length:** {max_len}"
    )

text_input = st.text_area(
    "Start typing a sentence...",
    value="the world as we",
    height=100,
    placeholder="e.g. it is our choices that show",
)

col1, col2 = st.columns(2)
with col1:
    generate_clicked = st.button("Generate continuation", type="primary", use_container_width=True)
with col2:
    suggest_clicked = st.button("Suggest next word", use_container_width=True)

if generate_clicked or suggest_clicked:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Predicting..."):
            generated_text, candidates = predict_next_words(text_input, num_words=num_words, top_k=top_k)

        if generate_clicked:
            st.subheader("Generated text")
            st.success(generated_text)

        st.subheader("Top next-word suggestions")
        if candidates:
            cols = st.columns(len(candidates))
            for c, (word, prob) in zip(cols, candidates):
                c.metric(label=word, value=f"{prob*100:.1f}%")
        else:
            st.info("Couldn't find any known words in your input — try different words.")

st.markdown("---")
st.caption(
    "Note: this model was trained on a fairly small quotes dataset, so predictions "
    "work best when your input resembles quote-like, literary phrasing."
)
