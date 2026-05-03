# 🚀 ReelIQ: Viral Video Success Predictor

This project is an AI-powered success predictor for short-form video content (Reels/TikTok). It predicts whether a video will have Low, Moderate, or Viral reach based purely on **pre-publish metadata** (hook strength, posting time, duration, etc.) without relying on post-publish data to prevent data leakage.

## 🏗️ Architecture
- **Machine Learning:** XGBoost Classifier (Trained via Google Colab)
- **Backend:** FastAPI (Python) serving the model predictions
- **Frontend:** Vanilla HTML/CSS/JS (Bento-box UI layout)

---

## 🛠️ Local Setup Instructions (For the Team)

If you are pulling this project to work on it locally, follow these steps on your Mac/Linux terminal:

**1. Clone the repository**
```bash
git clone [https://github.com/itspxsh/ViralVideoPrediction.git](https://github.com/itspxsh/ViralVideoPrediction.git)
cd ViralVideoPrediction
```

**2. Create and activate a Virtual Environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install the dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Local Server (FastAPI)**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**5. Test the UI**
Open the `index.html` file in your web browser. It will automatically connect to `http://localhost:8000` to fetch predictions.

---

## 🎯 Next Steps & Tasks for the Team
Here is what we need to focus on next before final deployment:

1. **Frontend Refinement (Vercel Prep):** 
   - Right now, `index.html` points to `http://localhost:8000/predict`.
   - **Task:** Once the backend is deployed to the cloud, we need to update the `fetch()` URL inside `index.html` to point to the real production API URL.
2. **Backend Deployment (Render/Railway):** 
   - Deploy the FastAPI application (`app.py`, `requirements.txt`, and `.pkl` files) to a cloud provider like Render or Railway. 
3. **CORS Configuration:**
   - In `app.py`, update `allow_origins=["*"]` to the specific Vercel domain once the frontend is live to secure the API.
4. **Model Retraining (Optional):**
   - If anyone wants to tweak the ML pipeline, upload the `.ipynb` file back to Google Colab, retrain it, download the new `xgb_model.pkl` and `preprocessor.pkl`, and push them to this repo.