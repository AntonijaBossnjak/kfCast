import numpy as np
import streamlit as st
from baseline import predict_ln_kf, predict_kf, C, A, A0

st.set_page_config(
    page_title="kfCast — Protein Folding Rate Predictor",
    page_icon="🧬",
)

st.title("🧬 kfCast")
st.markdown("##### Protein Folding Rate Predictor")

st.markdown(
    "This is a **baseline model** that uses only the length of the protein "
    "sequence (model: `ln(kf) = C + A·ln(L + a0)`)."
)

st.markdown("---")

st.markdown(
    "**Instructions:** paste a protein sequence below (letters only, e.g. "
    "`MKTAYIAKQRQISFVKSHFSRQ...`). Then click **Predict**."
)

EXAMPLE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Try an example"):
        st.session_state["seq"] = EXAMPLE

seq = st.text_area(
    "Protein sequence:",
    value=st.session_state.get("seq", ""),
    height=120,
    placeholder="e.g. MKTAYIAKQRQISFVKSHFSRQ...",
)

if st.button("Predict", type="primary"):
    seq_clean = seq.strip().upper()
    if not seq_clean:
        st.warning("Please enter a sequence before predicting.")
    elif not seq_clean.isalpha():
        st.error("Sequence may contain only letters (amino acids).")
    else:
        try:
            ln_kf = predict_ln_kf(seq_clean)
            kf = predict_kf(seq_clean)
        except ValueError as e:
            st.error(str(e))
        else:
            st.markdown("### Result")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sequence length", f"{len(seq_clean)} aa")
            c2.metric("kf (predicted)", f"{kf:.4g}")
            c3.metric("ln(kf)", f"{ln_kf:.4f}")

            with st.expander("Calculation details"):
                st.write(
                    f"Model: `ln(kf) = {C:.4f} + ({A:.4f})·ln(L + ({A0:.4f}))`"
                )
                st.write(f"L = {len(seq_clean)}")
                st.write(f"ln(L + a0) = {np.log(len(seq_clean) + A0):.4f}")
                st.write(f"ln(kf) = {ln_kf:.4f}")
                st.write(f"kf = exp(ln(kf)) = {kf:.4g}")

st.markdown("---")
st.caption(
    "kfCast · Baseline model trained on the S80_c1 dataset (51 proteins). "
    "Serves as a reference baseline for comparison with more complex models."
)