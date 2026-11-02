import numpy as np
import streamlit as st
from baseline import predict_ln_kf, predict_kf, get_model

st.set_page_config(
    page_title="kfCast — Protein Folding Rate Predictor",
    page_icon="🧬",
)

st.title("🧬 kfCast")
st.markdown("##### Protein Folding Rate Predictor")

st.markdown(
    "Predicts the protein folding rate constant **kf** from a protein sequence. "
    "Two baseline models are shown side by side for comparison."
)

st.markdown("---")

st.markdown(
    "**Instructions:** paste a protein sequence below (letters only, e.g. "
    "`MKTAYIAKQRQISFVKSHFSRQ...`). Then click **Predict**."
)

EXAMPLE = "MKAAGTRWVVHILLEYFDSSAAGGNMPPMKAAGTRWVVHILLEYFDSSAAGGNMPP"

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
            ln_kf_a0 = predict_ln_kf(seq_clean, "a0")
            kf_a0    = predict_kf(seq_clean, "a0")
            ln_kf_L  = predict_ln_kf(seq_clean, "L")
            kf_L     = predict_kf(seq_clean, "L")
        except ValueError as e:
            st.error(str(e))
        else:
            st.markdown(f"### Result  ·  sequence length = {len(seq_clean)} aa")
            st.markdown("---")

            m_a0 = get_model("a0")
            m_L  = get_model("L")

            col_a0, col_L = st.columns(2)

            with col_a0:
                st.markdown(f"#### Model 1 — {m_a0['name']}")
                st.markdown("`ln(kf) = C + A·ln(L + a0)`")
                st.metric("kf (predicted)", f"{kf_a0:.4g}")
                st.metric("ln(kf)", f"{ln_kf_a0:.4f}")
                with st.expander("Calculation details"):
                    st.write(f"C = {m_a0['C']:.6f}")
                    st.write(f"A = {m_a0['A']:.6f}")
                    st.write(f"a0 = {m_a0['a0']:.6f}")
                    st.write(f"ln(L + a0) = {np.log(len(seq_clean) + m_a0['a0']):.4f}")

            with col_L:
                st.markdown(f"#### Model 2 — {m_L['name']}")
                st.markdown("`ln(kf) = C + A·ln(L)`")
                st.metric("kf (predicted)", f"{kf_L:.4g}")
                st.metric("ln(kf)", f"{ln_kf_L:.4f}")
                with st.expander("Calculation details"):
                    st.write(f"C = {m_L['C']:.6f}")
                    st.write(f"A = {m_L['A']:.6f}")
                    st.write(f"ln(L) = {np.log(len(seq_clean)):.4f}")

            st.markdown("---")
            st.markdown("### Model parameters and performance")
            st.caption(
                f"Dataset: **S80_c1** · "
                f"Train set N₁ = {m_a0['N_train']} · "
                f"Test set N₂ = {m_a0['N_test']}"
            )

            rows = []
            labels = [
                ("C", "C"),
                ("A", "A"),
                ("a0", "a0"),
                ("RMSE_train", "RMSE (train)"),
                ("PCC_train", "PCC (train)"),
                ("RMSE_cv", "RMSE (LOOCV)"),
                ("PCC_cv", "PCC (LOOCV)"),
                ("RMSE_test", "RMSE (test)"),
                ("PCC_test", "PCC (test)"),
            ]
            for key, label in labels:
                v_a0 = m_a0.get(key)
                v_L  = m_L.get(key)
                rows.append({
                    "Quantity": label,
                    "Model 1 — ln(L + a0)": "—" if v_a0 is None else f"{v_a0:.4f}" if isinstance(v_a0, float) else v_a0,
                    "Model 2 — ln(L)":      "—" if v_L  is None else f"{v_L:.4f}"  if isinstance(v_L,  float) else v_L,
                })
            st.table(rows)

st.markdown("---")
st.caption(
    "kfCast · Baseline models trained on the S80_c1 dataset. "
    "Reference baseline for comparison with more complex descriptor-based models."
)