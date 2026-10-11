# Heart Disease Prediction System

**Assignment 2, Task 2** — AIC354 Machine Learning Fundamentals

Predicts the likely presence of heart disease from **4 patient attributes**
(`Sex`, `ChestPainType`, `FastingBloodSugar`, `ExerciseAngina`), using a
Support Vector Classifier. Built by adapting the instructor-provided
Titanic Passenger Survival Prediction notebook (`../1 - Titanic Survival
Prediction/Titanic_Passenger_Survival_Recommended.ipynb`) to a new dataset,
following the exact same steps (Import → Load → Understand → Preprocess →
Encode → Train → Test → Deploy → Feedback).

## Dataset

- **Source:** [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
  (Cleveland database, 303 patients, 14 original attributes, CC BY 4.0 license)
- **Raw file:** `processed.cleveland.data` (as downloaded from UCI)
- **Derived file:** `sample-data.csv` — only the 4 selected input attributes
  plus the binary target, in human-readable categorical form (mirrors the
  Titanic project's `sample-data.csv` layout)

### Why these 4 attributes

The assignment requires selecting exactly 4 of the 13 commonly-used
attributes. `Sex`, `ChestPainType`, `FastingBloodSugar`, and
`ExerciseAngina` were chosen because all four are **naturally categorical**
in the source data (like Titanic's PClass/Gender/Sibling/Embarked), which
let the Titanic notebook's Label Encoding approach be reused with no
structural changes — only the category lists and column names differ.

### Target

The original UCI `num` column (0 = no disease, 1-4 = increasing severity)
was collapsed into a binary `HeartDisease` (Yes/No) column, following the
UCI dataset's own documented convention for this exact simplification —
directly analogous to Titanic's binary `Survived` (Yes/No).

## Result

Support Vector Classifier accuracy on the held-out test set: **0.72**
(see the notebook's Step 9 feedback phase for a Logistic Regression
comparison, which reached 0.79 on the same split).

## Files

| File | Purpose |
|---|---|
| `Heart_Disease_Prediction.ipynb` | Main notebook — run top to bottom |
| `processed.cleveland.data` | Raw UCI dataset as downloaded |
| `sample-data.csv` | Derived dataset: 4 selected attributes + target |
| `sample-data-encoded*.csv`, `training/testing-data-encoded.csv`, `model-predictions.csv` | Intermediate files produced by running the notebook |
| `svc_trained_model.pkl` | Trained model, also used by `../streamlit-app-heart/app.py` |

## Running Locally

```bash
cd "2 - Heart Disease Prediction"
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r ../streamlit-app-heart/requirements.txt scikit-learn prettytable seaborn matplotlib jupyter
jupyter notebook Heart_Disease_Prediction.ipynb
```

⚠️ **Teaching demo only.** Only 4 of the 13 commonly-used clinical
attributes are used (per the assignment constraint) — this is not a
medical diagnostic tool.
