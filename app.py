"""
Transformer-Based Chatbot — Educational Streamlit App
-------------------------------------------------------
Demonstrates: Tokenization, Model Architecture, Next-Token Prediction,
Self-Attention visualization, Generation Parameters, Conversation Memory,
and a live Dashboard — all built on top of a pre-trained GPT-2 model.
"""

import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Transformer Chatbot", page_icon="🤖", layout="wide")

MODEL_NAME = "gpt2"  # 124M params — small enough to run on CPU


# ---------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Transformer model (GPT-2)...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",  # FIX: forces the model to actually compute and
                                       # return attention weights. Newer default attention
                                       # backends (sdpa / flash-attention) skip this even
                                       # when output_attentions=True is requested, which
                                       # made out.attentions come back as an empty tuple
                                       # and caused: IndexError: tuple index out of range
    )
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()
config = model.config

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"role": ..., "text": ...}

if "stats" not in st.session_state:
    st.session_state.stats = {
        "questions": 0,
        "tokens_generated": 0,
        "response_times": [],
    }

if "last_input_tokens" not in st.session_state:
    st.session_state.last_input_tokens = None  # (tokens, ids) of the last user turn


# ---------------------------------------------------------------------------
# Sidebar: Model Info + Generation Settings + Dashboard
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🧩 Model Info")
    st.write(f"**Model name:** `{MODEL_NAME}`")
    st.write(f"**Transformer layers:** {config.n_layer}")
    st.write(f"**Attention heads:** {config.n_head}")
    st.write(f"**Embedding dimension:** {config.n_embd}")
    st.write(f"**Vocabulary size:** {config.vocab_size:,}")

    st.divider()
    st.header("🎛️ Generation Settings")
    temperature = st.slider(
        "Temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.1,
        help="Higher = more random/creative. Lower = more focused/deterministic."
    )
    top_k = st.slider(
        "Top-K", min_value=1, max_value=100, value=50, step=1,
        help="Model only samples from the K most likely next tokens."
    )
    max_tokens = st.slider(
        "Max new tokens", min_value=10, max_value=200, value=60, step=10,
        help="Maximum length of the generated response."
    )

    st.divider()
    st.header("📊 Dashboard")
    stats = st.session_state.stats
    turns = len(st.session_state.history) // 2

    col1, col2 = st.columns(2)
    col1.metric("Questions asked", stats["questions"])
    col2.metric("Conversation turns", turns)
    col1.metric("Tokens generated", stats["tokens_generated"])
    avg_time = np.mean(stats["response_times"]) if stats["response_times"] else 0.0
    col2.metric("Avg response time", f"{avg_time:.2f}s")

    st.caption(
        f"Current params → Temperature: **{temperature}**, "
        f"Top-K: **{top_k}**, Max tokens: **{max_tokens}**"
    )

    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.session_state.last_input_tokens = None
        st.rerun()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_chat, tab_tokens, tab_predict, tab_attention, tab_arch = st.tabs(
    ["💬 Chat", "🔤 Tokenization", "🔮 Next-Token Prediction", "🧠 Self-Attention", "ℹ️ Architecture"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Chat
# ---------------------------------------------------------------------------
with tab_chat:
    st.title("🤖 Transformer Chatbot")
    st.caption("Built on GPT-2 — a decoder-only Transformer — with full conversation memory.")

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.write(turn["text"])

    user_input = st.chat_input("Type your question...")

    if user_input:
        st.session_state.history.append({"role": "user", "text": user_input})

        # Build context from conversation memory (last 3 turns = 6 messages)
        context = ""
        for turn in st.session_state.history[-6:]:
            prefix = "User: " if turn["role"] == "user" else "AI: "
            context += prefix + turn["text"] + "\n"
        context += "AI:"

        inputs = tokenizer(context, return_tensors="pt")

        # Save tokenization of this turn's context for the other tabs
        input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        input_ids = inputs["input_ids"][0].tolist()
        st.session_state.last_input_tokens = (input_tokens, input_ids)

        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - start

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # GPT-2 isn't chat-tuned, so it may try to keep writing "User:" itself — cut it off
        response = response.split("User:")[0].split("AI:")[0].strip()
        if not response:
            response = "(model produced an empty response — try raising Temperature or Max tokens)"

        st.session_state.history.append({"role": "assistant", "text": response})

        stats["questions"] += 1
        stats["tokens_generated"] += len(generated_ids)
        stats["response_times"].append(elapsed)

        st.rerun()

# ---------------------------------------------------------------------------
# TAB 2 — Tokenization
# ---------------------------------------------------------------------------
with tab_tokens:
    st.header("🔤 From Text to Tokens to Token IDs")
    st.markdown(
        "Transformer models don't understand raw text — they operate on numbers. "
        "**Tokenization** breaks text into sub-word pieces (*tokens*), and each token "
        "is mapped to a unique **Token ID** from the model's fixed vocabulary."
    )

    demo_text = st.text_input("Try tokenizing any text:", "Machine learning is amazing!")

    if demo_text:
        tokens = tokenizer.tokenize(demo_text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        df = pd.DataFrame({"Token": tokens, "Token ID": ids})
        st.dataframe(df, use_container_width=True)

        st.info(
            "Notice how some words split into multiple tokens (e.g. rare words), "
            "and note the 'Ġ' symbol GPT-2 uses to mark a token that starts with a space. "
            "The Token IDs are what actually get fed into the model's embedding layer."
        )

    if st.session_state.last_input_tokens:
        st.subheader("Tokenization of the last chat context")
        toks, ids = st.session_state.last_input_tokens
        st.dataframe(pd.DataFrame({"Token": toks, "Token ID": ids}), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 — Next-Token Prediction
# ---------------------------------------------------------------------------
with tab_predict:
    st.header("🔮 Next-Token Prediction")
    st.markdown(
        "A Transformer language model generates text **one token at a time**: given the "
        "tokens so far, it outputs a probability distribution over the *entire vocabulary* "
        "for what comes next. Generation then repeats this — feeding the new token back in — "
        "until it stops."
    )

    demo_text2 = st.text_input(
        "Enter a prompt and see the top predicted next tokens:",
        "The capital of France is"
    )

    if demo_text2:
        inputs2 = tokenizer(demo_text2, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs2).logits
        next_token_logits = logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 10)
        top_tokens = [tokenizer.decode([idx]) for idx in top_indices]

        pred_df = pd.DataFrame({"Token": top_tokens, "Probability": top_probs.numpy()})
        st.bar_chart(pred_df.set_index("Token"))
        st.dataframe(pred_df, use_container_width=True)

        st.info(
            "**Temperature** reshapes this probability distribution before sampling "
            "(higher = flatter/more random, lower = sharper/more confident). "
            "**Top-K** restricts sampling to only the K highest-probability tokens shown above."
        )

# ---------------------------------------------------------------------------
# TAB 4 — Self-Attention
# ---------------------------------------------------------------------------
with tab_attention:
    st.header("🧠 Self-Attention Visualization")
    st.markdown(
        "Self-Attention lets every token look at every other token in the input and decide "
        "*how much* to focus on it when building its representation. Each layer has multiple "
        "**attention heads**, each potentially capturing a different kind of relationship "
        "(e.g. grammar, coreference, topic)."
    )

    if st.session_state.last_input_tokens:
        tokens, ids = st.session_state.last_input_tokens
        input_ids_t = torch.tensor([ids])

        with torch.no_grad():
            out = model(input_ids_t, output_attentions=True)
        attentions = out.attentions  # tuple: (n_layers) of (batch, heads, seq, seq)

        # FIX: guard against an empty/unexpected attentions tuple instead of
        # crashing with "IndexError: tuple index out of range"
        if not attentions or len(attentions) == 0 or attentions[0].dim() < 4:
            st.error(
                "⚠️ Couldn't retrieve attention weights from the model. "
                "This usually means the model didn't compute attentions for this backend. "
                "Try restarting the app, or ask a new question in the 💬 Chat tab first."
            )
        else:
            c1, c2 = st.columns(2)
            layer_idx = c1.slider("Layer", 0, len(attentions) - 1, 0)
            head_idx = c2.slider("Attention head", 0, attentions[0].shape[1] - 1, 0)

            attn_matrix = attentions[layer_idx][0, head_idx].numpy()

            MAX_SHOW = 25
            show_tokens = tokens[:MAX_SHOW]
            attn_show = attn_matrix[:MAX_SHOW, :MAX_SHOW]

            fig, ax = plt.subplots(figsize=(7, 7))
            im = ax.imshow(attn_show, cmap="viridis")
            ax.set_xticks(range(len(show_tokens)))
            ax.set_yticks(range(len(show_tokens)))
            ax.set_xticklabels(show_tokens, rotation=90, fontsize=8)
            ax.set_yticklabels(show_tokens, fontsize=8)
            ax.set_xlabel("Attending TO (key)")
            ax.set_ylabel("Attending FROM (query)")
            fig.colorbar(im, fraction=0.046, pad=0.04)
            st.pyplot(fig)

            st.info(
                "Each row shows how much that token 'attends to' every other token (columns). "
                "Brighter cells = stronger attention weight. Try different layers/heads — "
                "early layers often capture local/syntactic patterns, later layers more semantic ones."
            )
    else:
        st.warning("Ask the chatbot a question in the 💬 Chat tab first, so there's a context to visualize.")

# ---------------------------------------------------------------------------
# TAB 5 — Architecture explanation
# ---------------------------------------------------------------------------
with tab_arch:
    st.header("ℹ️ System Architecture")
    st.markdown(f"""
**Pipeline overview:**

1. **User input** is typed in the chat box.
2. **Tokenizer** splits the text (plus prior conversation turns, for memory) into sub-word tokens
   and converts them into Token IDs — GPT-2's vocabulary has **{config.vocab_size:,}** entries.
3. **Embedding layer** maps each Token ID to a **{config.n_embd}**-dimensional vector, and adds
   positional information so the model knows token order.
4. **Transformer decoder stack**: the embeddings pass through **{config.n_layer}** stacked decoder
   blocks. Each block has:
   - **Multi-head Self-Attention** ({config.n_head} heads) — tokens exchange information with each other.
   - **Feed-forward network** — further transforms each token's representation.
   - Residual connections + layer normalization around both.
5. **Output head** projects the final hidden state back to a **{config.vocab_size:,}**-dimensional
   vector of logits — one score per possible next token.
6. **Sampling** (Temperature + Top-K) turns those logits into a probability distribution and picks
   the next token.
7. This new token is appended to the sequence and **steps 3-6 repeat** until `Max new tokens`
   is reached or an end-of-text token is produced.
8. The full generated text is decoded back into a string and shown as the AI's reply.
9. **Conversation memory**: previous turns are re-included as text context in step 2 of every
   new request — GPT-2 has no built-in persistent memory, so the app re-feeds the recent
   history each time.

**Why GPT-2 specifically?** It's a *decoder-only* Transformer — the same family of architecture
used by modern chat LLMs — which makes it ideal for demonstrating causal (left-to-right)
next-token generation, self-attention, and tokenization concepts in a lightweight package that
runs on CPU.
""")
