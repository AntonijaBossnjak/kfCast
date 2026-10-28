# kfCast

**Protein Folding Rate Predictor**

kfCast is a web application for predicting the protein folding rate constant
(`kf`) directly from a protein sequence. This first version implements a
**baseline model** that uses only the protein sequence length.

## Model

The baseline model is a log-log linear regression:

ln(kf) = C + A · ln(L + a0)


where:

- `L` — protein length (number of amino acids)
- `C`, `A`, `a0` — fitted parameters

Fitted parameters (on the S80_c1 TRAIN set, 51 proteins):

| Parameter | Value |
|-----------|-------|
| C         | 34.1764 |
| A         | -6.8288 |
| a0        | -3.0057 |

**Performance on the independent TEST set (29 proteins):**

- RMSE = 2.593
- PCC = 0.847

This baseline serves as a reference for comparison with more complex
descriptor-based models.

## Usage

### Web app

Once deployed, the app is available at: *(link will be added after deployment)*

Paste a protein sequence (letters only, e.g. `MKTAYIAKQRQISFVKSHFSRQ...`)
and click **Predict** to obtain the predicted `kf` and `ln(kf)`.

### Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

Project structure
text

kfCast/
├── app.py            # Streamlit web interface
├── baseline.py       # Baseline model and fitted parameters
├── requirements.txt  # Python dependencies
└── README.md

Requirements

    Python 3.10+

    streamlit

    numpy

Notes

    The model requires sequences with L > 3 (due to the a0 = -3.006 offset).

    Sequences are validated: only alphabetic characters (amino acids) are accepted.

    This is a baseline model — future versions will include descriptor-based
    models (secondary structure, hydrophobicity, etc.) for improved accuracy.

Author

Antonija Kraljević