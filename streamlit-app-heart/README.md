# Heart Disease Predictor — Streamlit App

A small web app to try the trained Heart Disease model in a browser, no
install required. Built from the model trained in
[`../2 - Heart Disease Prediction`](../2%20-%20Heart%20Disease%20Prediction)
(the `Heart_Disease_Prediction.ipynb` notebook). Structured identically to
`../streamlit-app-titanic`.

## Deploying to Streamlit Community Cloud (free)

1. Push this whole repo to a **public GitHub repo** under your own account.
2. Go to **https://share.streamlit.io** and sign in with your GitHub account.
3. Click **"Create app"** → **"Deploy a public app from GitHub"**.
4. Fill in:
   - **Repository:** `<your-github-username>/<your-repo-name>`
   - **Branch:** `main`
   - **Main file path:** `streamlit-app-heart/app.py`
5. Click **Deploy**. The first deploy takes a minute or two while it installs `requirements.txt`.
6. You'll get a public URL like `https://<something>.streamlit.app`.
7. Paste that URL into the **Live Deployment Link** field at the top of
   `../2 - Heart Disease Prediction/Heart_Disease_Prediction.ipynb`, per the
   assignment's submission guidelines.

## Running It Locally First (optional, to test changes)

```bash
cd streamlit-app-heart
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

This opens the app at `http://localhost:8501`.

## How It Works

- `app.py` loads `svc_trained_model.pkl` (the same file the notebook
  produces) and recreates the exact same `LabelEncoder`s used during
  training, so predictions match the notebook exactly.
- Users pick Sex, Chest Pain Type, Fasting Blood Sugar, and Exercise-Induced
  Angina from dropdowns and click **Predict**.
- If you retrain the model in the notebook and want the app to reflect it,
  copy the new `svc_trained_model.pkl` into this folder and push — Streamlit
  Cloud will auto-redeploy.
