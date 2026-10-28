import numpy as np

# ============================================================
# Baseline model: ln(kf) = C + A * ln(L + a0)
# Fit on the S80_c1 TRAIN set (51 proteins)
# ============================================================

C  = 34.1764375053294
A  = -6.82879121043065
A0 = -3.00566374696263


def predict_ln_kf(sequence: str) -> float:
    """Return predicted ln(kf) for a protein sequence."""
    seq = sequence.strip().upper()
    L = len(seq)
    if L == 0:
        raise ValueError("Sequence is empty.")
    inside = L + A0
    if inside <= 0:
        raise ValueError(
            f"Sequence is too short for this model "
            f"(length {L}, model requires L > {abs(A0):.2f})."
        )
    return C + A * np.log(inside)


def predict_kf(sequence: str) -> float:
    """Return predicted folding rate constant kf."""
    return float(np.exp(predict_ln_kf(sequence)))