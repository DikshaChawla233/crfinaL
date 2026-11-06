# Career Lens — Ready-to-Deploy Streamlit App

This folder contains the Streamlit frontend and the three model artifacts exported from your Career Lens notebook.

## Files
- `app.py` — Streamlit frontend
- `requirements.txt` — Python dependencies (scikit-learn is pinned to 1.6.1, the version used to save the model)
- `career_lens_model_compressed.pkl` — trained Random Forest regression model
- `career_lens_scaler.pkl` — fitted StandardScaler
- `career_lens_features.pkl` — exact 78 training feature names
- `README.md` — these instructions

## Run locally
1. Extract the ZIP file.
2. Install Python 3.10 or 3.11.
3. Open a terminal in the extracted folder and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

4. Open the local URL printed in the terminal (usually `http://localhost:8501`).

## Deploy online with Streamlit Community Cloud
1. Create a GitHub repository named `career-lens`.
2. Upload all six files from this folder to the repository.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
5. Click **Deploy** and wait for installation to finish.
6. Open the app URL and test it with a sample resume.

The model is saved with compression so the individual model file stays below GitHub’s 25 MiB browser-upload limit. Do not upload your original dataset or real people's resumes unless you have permission and a clear privacy reason.

## Model and prototype limitations
- The saved Random Forest model expects 78 features. The app builds those columns in the exact order saved in `career_lens_features.pkl` and applies the saved scaler before prediction.
- Resume skill extraction is keyword-based and is not a full NLP resume parser.
- Scanned/image-only PDFs may not yield text.
- Some job-title or education categories entered by a user may not exactly match the training categories, which can affect score reliability.
- The score is an estimate, not the probability of being hired. Review results manually.
