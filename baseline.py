import numpy as np

# ============================================================
# kfCast — baseline models
# Dataset: S80_c1 (N_train = 51, N_test = 29)
# ============================================================

MODEL_A0 = {
    "name": "ln(L + a0)",
    "C": 34.1764375053294,
    "A": -6.82879121043065,
    "a0": -3.00566374696263,
    "N_train": 51,
    "N_test": 29,
    "RMSE_train": 2.7048,
    "PCC_train": 0.7689,
    "RMSE_cv": 2.8260,
    "PCC_cv": 0.7449,
    "RMSE_test": 2.5932,
    "PCC_test": 0.8471,
}

MODEL_L = {
    "name": "ln(L)",
    "C": 35.6832357035556,
    "A": -7.1062253718281,
    "a0": None,
    "N_train": 51,
    "N_test": 29,
    "RMSE_train": 2.7052,
    "PCC_train": 0.7688,
    "RMSE_cv": 2.7975,
    "PCC_cv": 0.7505,
    "RMSE_test": 2.5505,
    "PCC_test": 0.8482,
}


def _predict_ln_kf(sequence: str, model: dict) -> float:
    seq = sequence.strip().upper()
    L = len(seq)
    if L == 0:
        raise ValueError("Sequence is empty.")
    inside = L + (model["a0"] if model["a0"] is not None else 0.0)
    if inside <= 0:
        raise ValueError(
            f"Sequence is too short for model '{model['name']}' "
            f"(length {L}, requires L > {abs(model['a0']):.2f})."
        )
    return model["C"] + model["A"] * np.log(inside)


def predict_ln_kf(sequence: str, model_key: str = "a0") -> float:
    model = MODEL_A0 if model_key == "a0" else MODEL_L
    return _predict_ln_kf(sequence, model)


def predict_kf(sequence: str, model_key: str = "a0") -> float:
    return float(np.exp(predict_ln_kf(sequence, model_key)))


def get_model(model_key: str) -> dict:
    return MODEL_A0 if model_key == "a0" else MODEL_L