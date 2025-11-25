# Urban Farmer — AI Crop Recommendation & Smart Farming Assistant

A practical, end-to-end AI assistant that helps growers choose the right crops for their soil and climate, evaluate feasibility, and act on weather-driven irrigation guidance. Built with Streamlit, scikit-learn, LangChain + Gemini, and OpenWeather.

## 🚀 Elevator Pitch
- Make crop selection and farm decisions data-driven, not guesswork.
- Combine soil nutrients, local weather, and agronomic knowledge to recommend viable crops and corrective actions.
- Deliver actionable guidance: irrigation amounts, risk alerts, and season-aware tips.
- Provide a friendly chat assistant with RAG that explains “why” behind every recommendation.

## 🌱 Impact
- Reduce wasted inputs by aligning N/P/K and pH to crop needs.
- Improve yields by selecting crops suited to current soil and climate.
- Cut water usage with weather-aware irrigation scheduling and alerts.
- Empower smallholders with accessible, explainable recommendations and practical steps.

## 🧩 Key Features
- AI-powered crop recommendations from soil and climate inputs (N/P/K, temperature, humidity, pH, rainfall).
- Feasibility analysis: parameter-by-parameter comparison against ideal crop conditions with corrective actions.
- Chat Assistant (Gemini via LangChain + FAISS RAG) that answers agronomy questions using dataset-derived context.
- Weather integration (OpenWeather): current conditions, 15-day forecast, heat/frost alerts.
- Irrigation guidance and proactive notifications via email and WhatsApp (Twilio).
- Visuals: forecast charts, compatibility scores, nutrient comparisons.

## ⚙️ Architecture
- `urban_farmer.py`: Streamlit application integrating
  - OpenWeather API for current + forecast data
  - AI recommendations and feasibility analysis (LLM: Gemini)
  - RAG over `crop_context.txt` using `sentence-transformers` + FAISS
  - Email (SMTP) and WhatsApp (Twilio) alerts
- `test.py`: Lightweight Streamlit demo that trains a `RandomForestClassifier` on the dataset and exposes
  - Top-3 crop predictions, feasibility checks
  - Compatibility score, nutrient & environmental alerts
  - Chatbot with RAG over `crop_context.txt`
- `Crop_Recommendation.csv`: Dataset of soil & climate features with labeled crops.
- `crop_prediction.ipynb`: Notebook for EDA/model experimentation and visualizations.
- `crop_context.txt`: Auto-generated knowledge base used by the chat/RAG.
- Model artifacts: `crop_recommendation_model.joblib`, `label_encoder.joblib`, `minmax_scaler.joblib` (optional pipeline assets).

## 🧠 Data & Modeling
- Features: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature (°C), Humidity (%), pH, Rainfall (mm).
- Model: `RandomForestClassifier` for crop recommendation (in `test.py`).
- Knowledge: Ideal ranges computed per crop from the dataset; exported to `crop_context.txt` for retrieval-augmented answers.
- Explainability: LLM prompts compare current vs. ideal parameters and propose targeted corrections.

## 📸 App Walkthrough
- Input soil and climate values in Streamlit.
- Get top crop candidates with reasons and minor adjustments.
- See feasibility analysis for a selected crop (tabular comparison + actions).
- Visualize weather and receive irrigation guidance & risk alerts (heat/frost/heavy rain).
- Chat with the assistant; answers are grounded in `crop_context.txt`.

## 🛠️ Setup
Requires Python 3.9+.

Recommended packages:
```
streamlit
pandas
numpy
scikit-learn
seaborn
matplotlib
requests
google-generativeai
langchain
langchain-community
sentence-transformers
faiss-cpu
twilio
```
Install in PowerShell:

```powershell
# (Optional) Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install packages
pip install -U pip
pip install streamlit pandas numpy scikit-learn seaborn matplotlib requests `
  google-generativeai langchain langchain-community sentence-transformers faiss-cpu twilio
```

## 🔐 Secrets & Configuration
Move hard-coded keys to environment variables or Streamlit secrets.

Example `~/.streamlit/secrets.toml`:
```toml
OPENWEATHER_API_KEY = "<your-key>"
GOOGLE_API_KEY = "<your-gemini-key>"
EMAIL_SENDER = "<your-email>"
EMAIL_PASSWORD = "<your-app-password>"
EMAIL_RECEIVER = "<default-notify-email>"
TWILIO_SID = "<your-twilio-sid>"
TWILIO_AUTH_TOKEN = "<your-twilio-token>"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+1xxxxxxxxxx"
```
Then in code, read via `st.secrets[...]` or `os.environ[...]`.

## ▶️ Run It
Run the full assistant:
```powershell
streamlit run urban_farmer.py
```
Try the simpler demo:
```powershell
streamlit run test.py
```
Optional notebook (EDA/experiments): open `crop_prediction.ipynb` in VS Code and run cells.

## 🧪 Recruiter-Friendly Demos
- Change N/P/K and pH to see dynamic recommendations and corrective actions.
- Switch city to observe forecast-based irrigation guidance and alerts.
- Ask the chat assistant “Why is Maize suitable?” or “How to adjust soil for Rice?” — answers cite dataset context.

## 📁 Repository Structure
```
├── urban_farmer.py           # Streamlit app (LLM + weather + alerts + RAG)
├── test.py                   # Streamlit demo with RandomForest + RAG chatbot
├── Crop_Recommendation.csv   # Dataset
├── crop_prediction.ipynb     # EDA / training notebook
├── crop_context.txt          # Auto-generated crop knowledge base
├── crop_recommendation_model.joblib
├── label_encoder.joblib
├── minmax_scaler.joblib
└── README.md
```

## 📈 Future Work
- Secure secrets via `st.secrets` and remove keys from source.
- Integrate the saved model pipeline end-to-end in `urban_farmer.py`.
- Add GIS layers (soil maps) and farm location to personalize recommendations.
- Add localization and accessibility improvements.
- Benchmark model variants and report metrics.

## 🙌 Acknowledgements
- Dataset: Crop Recommendation (public agronomy dataset).
- LLM: Google Gemini via `google-generativeai`.
- NLP/RAG: LangChain + `sentence-transformers` + FAISS.
- Weather: OpenWeather API.

---
If you'd like, I can also add a `requirements.txt` and wire secrets loading into the app for production readiness.
