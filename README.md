# AIC354 — Assignment 2: Developing an End-to-End Machine Learning System

**Student:** Mujtaba Naveed | FA24-BSE-183
**Course:** AIC354 — Machine Learning Fundamentals, Fall 2026
**Instructor:** Dr. Rao Muhammad Adeel Nawab

This repo contains both tasks of Assignment 2:

| Task | Folder | Live App |
|---|---|---|
| **Task 1** — Titanic Passenger Survival Prediction (instructor-provided; run, understand, deploy) | [`1 - Titanic Survival Prediction`](./1%20-%20Titanic%20Survival%20Prediction) | [`streamlit-app-titanic`](./streamlit-app-titanic) — _PASTE YOUR DEPLOYED URL HERE_ |
| **Task 2** — Heart Disease Prediction (adapted from Task 1's code, per assignment instructions) | [`2 - Heart Disease Prediction`](./2%20-%20Heart%20Disease%20Prediction) | [`streamlit-app-heart`](./streamlit-app-heart) — _PASTE YOUR DEPLOYED URL HERE_ |

Both follow the same teaching structure:
**Import Libraries → Load Data → Understand & Preprocess → Label Encode →
Train → Test → Application Phase → Feedback Phase → Deploy.**

## Task 1 — Titanic Survival Prediction

Instructor-provided code (unmodified) — predicts survival from PClass,
Gender, Sibling count, and Embarked port using an SVC. **Your job for Task
1** is to run it locally yourself, read through it so you actually
understand each step, and deploy your own copy to Streamlit Community
Cloud (see "Deploying" below) — don't just rely on the instructor's own
live demo link, since the assignment asks you to deploy it yourself.

## Task 2 — Heart Disease Prediction

Built by adapting Task 1's exact pipeline to the
[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
(Cleveland database), using 4 selected input attributes (`Sex`,
`ChestPainType`, `FastingBloodSugar`, `ExerciseAngina`) as required by the
assignment. See [`2 - Heart Disease Prediction/README.md`](./2%20-%20Heart%20Disease%20Prediction/README.md)
for full details on attribute selection and results (SVC: 0.72 accuracy).

## Deploying to Streamlit Community Cloud

Both apps deploy the same way:

1. Push this repo to **your own public GitHub account**.
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. **Create app** → **Deploy a public app from GitHub**.
4. Repository: `<your-username>/<this-repo>`, Branch: `main`, Main file
   path: `streamlit-app-titanic/app.py` (for Task 1) or
   `streamlit-app-heart/app.py` (for Task 2).
5. Deploy — you'll need to do this **twice**, once per app, since each is a
   separate Streamlit deployment pointing at a different main file in the
   same repo.
6. Copy each resulting URL into the matching notebook's **Live Deployment
   Link** field at the top, as the assignment requires.

Full step-by-step instructions (with troubleshooting) are in each
`streamlit-app-*/README.md`.

## Running Everything Locally

Each project folder is self-contained, with its own `requirements.txt`
(deliberately **not** a shared root-level file — a single root
`requirements.txt` can confuse Streamlit Cloud's dependency lookup when a
repo contains multiple apps).

```bash
# For a Streamlit app:
cd streamlit-app-titanic   # or streamlit-app-heart
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

# For a notebook:
cd "1 - Titanic Survival Prediction"   # or "2 - Heart Disease Prediction"
pip install pandas numpy scikit-learn matplotlib seaborn prettytable jupyter
jupyter notebook
```

## Submission Checklist

- [ ] Both notebooks run top-to-bottom with zero errors (verified before
      upload — see each notebook's own saved outputs)
- [ ] Both Streamlit apps deployed and live under your own account
- [ ] Both notebooks have their **Live Deployment Link** filled in at the
      top
- [ ] Code, dataset(s), and trained model file(s) uploaded to Google
      Classroom per the assignment's platform instructions
- [ ] Submitted before Sunday, October 11, 2026, 23:59
