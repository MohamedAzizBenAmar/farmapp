import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Initialisation Streamlit (doit être la première commande)
st.set_page_config(layout="wide", page_title="🌾 Smart Farming Assistant")

# Chargement sécurisé des variables d'environnement
load_dotenv()

def get_config_value(key, default=None, replace_underscores=False):
    """Récupère une valeur de configuration de manière sécurisée"""
    value = os.getenv(key, default)
    if value is None:
        st.error(f"Configuration manquante: {key}")
        return None
    if replace_underscores and isinstance(value, str):
        return value.replace("_", " ")
    return value

# Configuration des clés API
API_KEY = get_config_value("OPENWEATHER_API_KEY")
GOOGLE_API_KEY = get_config_value("GOOGLE_API_KEY")
EMAIL_SENDER = get_config_value("EMAIL_SENDER")
EMAIL_PASSWORD = get_config_value("EMAIL_PASSWORD", replace_underscores=True)
EMAIL_RECEIVER = get_config_value("EMAIL_RECEIVER")
TWILIO_SID = get_config_value("TWILIO_SID")
TWILIO_AUTH_TOKEN = get_config_value("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = get_config_value("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
RECEIVER_WHATSAPP = get_config_value("RECEIVER_WHATSAPP", "whatsapp:+21626720354")

# Vérification des clés essentielles
if not all([API_KEY, GOOGLE_API_KEY]):
    st.error("Configuration API manquante. Vérifiez vos clés dans le fichier .env")
    st.stop()

# Initialisation des services
genai.configure(api_key=GOOGLE_API_KEY)

# ================ CONSTANTES ET CONFIGURATION ================
CITY = "Toronto"
CROP = "maize"

# Crop-specific weekly watering needs in mm (1mm = 1L/m²)
watering_guide = {
    # Cereals
    "rice": 40,        # Requires flooded conditions
    "wheat": 25,       # Moderate water needs
    "maize": 30,       # Moderate-high water needs
    "barley": 20,      # Drought tolerant
    
    # Pulses/Legumes
    "chickpea": 20,    # Drought tolerant
    "kidneybeans": 25, # Moderate water needs
    "pigeonpeas": 18,  # Drought tolerant
    "mothbeans": 20,   # Drought tolerant
    "mungbean": 22,    # Moderate water needs
    "blackgram": 23,   # Moderate water needs
    "lentil": 18,      # Drought tolerant
    
    # Fruits
    "mango": 35,       # Needs regular watering when young
    "grapes": 30,      # Regular watering needed
    "watermelon": 40,  # High water needs
    "muskmelon": 35,   # High water needs
    "apple": 30,       # Regular watering needed
    "orange": 32,      # Regular watering needed
    "papaya": 35,      # Needs consistent moisture
    "coconut": 30,     # Regular watering in non-humid climates
    
    # Commercial Crops
    "cotton": 35,      # High water needs
    "jute": 40,        # Requires moist conditions
    "coffee": 28       # Regular watering needed
}

# Email config (you should move these to secrets.toml)

# WhatsApp config (you should move these to secrets.toml)


# ------------------ CONFIG ------------------
genai.configure(api_key=GOOGLE_API_KEY)

# ------------------ DATA LOADING ------------------
@st.cache_data
def load_data():
    data = pd.read_csv("Crop_Recommendation.csv")
    ideal_conditions = data.groupby("Crop").mean()
    ranges = data.groupby("Crop").agg(["min", "max"])
   
    # Generate context file for RAG
    context_data = ""
    for crop in data["Crop"].unique():
        ideal = ideal_conditions.loc[crop]
        context_data += f"Culture : {crop}\n"
        context_data += f"Conditions idéales - N: {ideal['Nitrogen']:.1f}, P: {ideal['Phosphorus']:.1f}, K: {ideal['Potassium']:.1f}, Temp: {ideal['Temperature']:.1f}°C, Hum: {ideal['Humidity']:.1f}%, pH: {ideal['pH_Value']:.1f}, Rain: {ideal['Rainfall']:.1f} mm\n\n"
   
    with open("crop_context.txt", "w") as f:
        f.write(context_data)
   
    return data, ideal_conditions, ranges

data, ideal_conditions, ranges = load_data()
required_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


def get_15_day_forecast(city):
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    # Process forecast data
    forecast_list = data["list"]
    daily_data = {}

    for forecast in forecast_list:
        date = forecast["dt_txt"].split()[0]
        if date not in daily_data:
            daily_data[date] = {
                "temps": [],
                "rain": [],
                "humidity": [],
                "conditions": []
            }

        daily_data[date]["temps"].append(forecast["main"]["temp"])
        daily_data[date]["rain"].append(forecast.get("rain", {}).get("3h", 0))
        daily_data[date]["humidity"].append(forecast["main"]["humidity"])
        daily_data[date]["conditions"].append(forecast["weather"][0]["main"])

    # Aggregate to daily values
    processed_data = []
    for date, values in daily_data.items():
        processed_data.append({
            "date": date,
            "avg_temp": np.mean(values["temps"]),
            "max_temp": max(values["temps"]),
            "min_temp": min(values["temps"]),
            "total_rain": sum(values["rain"]),
            "avg_humidity": np.mean(values["humidity"]),
            "main_condition": max(set(values["conditions"]), key=values["conditions"].count)
        })

    return processed_data[:15]  # Return only next 15 days
def evaluate_weather_impact(crop, forecast_data):
    alerts = []
    recommendations = []

    for day in forecast_data:
        water_needed = watering_guide.get(crop, 30)

        # Adjust based on rainfall
        if day["total_rain"] > 50:
            water_needed *= 0.5
            recommendations.append(f"{day['date']}: Reduce watering by 50% due to heavy rain")
        elif day["total_rain"] < 10:
            water_needed *= 1.2
            recommendations.append(f"{day['date']}: Increase watering by 20% due to dry conditions")

        # Check for extreme temperatures
        if day["max_temp"] > 35:
            alerts.append(f"🌡️ Heatwave alert on {day['date']} (Max: {day['max_temp']}°C)")
            recommendations.append(f"{day['date']}: Water in early morning/late evening to prevent evaporation")
        elif day["min_temp"] < 5:
            alerts.append(f"❄️ Frost alert on {day['date']} (Min: {day['min_temp']}°C)")
            recommendations.append(f"{day['date']}: Use row covers to protect crops from frost")

    return alerts, recommendations

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    main = data.get("main", {})
    wind = data.get("wind", {})
    weather = data.get("weather", [{}])[0]
    rain = data.get("rain", {}).get("1h", 0)

    return {
        "temperature": main.get("temp"),
        "humidity": main.get("humidity"),
        "wind_speed": wind.get("speed"),
        "rainfall": rain,
        "condition": weather.get("main", "")
    }

def get_weather_forecast(city):
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    forecast_list = data["list"]

    alerts = []
    for slot in forecast_list:
        temp = slot["main"]["temp"]
        rain = slot.get("rain", {}).get("3h", 0)
        dt_txt = slot["dt_txt"]

        if temp > 35:
            alerts.append(f"🔥 Heatwave expected on {dt_txt}")
        if rain > 50:
            alerts.append(f"🌧️ Heavy rain expected on {dt_txt}")

    return alerts

def evaluate_conditions(crop, weather):
    temperature = weather["temperature"]
    rainfall = weather["rainfall"]
    condition = weather["condition"]

    water_needed = watering_guide.get(crop, 30)

    if rainfall > 50:
        water_needed *= 0.5
    elif rainfall < 10:
        water_needed *= 1.2

    if temperature > 35:
        advice = "🔥 Heat alert: Use mulch and water in early morning or late evening."
    elif condition.lower() == "rain":
        advice = "🌧️ Rain expected: Ensure drainage and avoid overwatering."
    elif temperature < 5:
        advice = "❄️ Cold alert: Use row covers or tunnels to protect crops."
    else:
        advice = "✅ Conditions are normal. Maintain regular irrigation schedule."

    return {
        "crop": crop,
        "recommended_water_l_m2": round(water_needed, 2),
        "weather": weather,
        "advice": advice
    }

def send_email(subject, body,mail):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = mail

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, [EMAIL_RECEIVER], msg.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

def send_whatsapp(message,whatsapp):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message,
            to=whatsapp
        )
        return True
    except Exception as e:
        st.error(f"Failed to send WhatsApp message: {e}")
        return False


class GeminiLLM(Runnable):
    def __init__(self, model_name="gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name)
        
    def invoke(self, input: dict, config=None, **kwargs):
        # Handle both direct prompt and LangChain formatted input
        if isinstance(input, dict):
            prompt = input.get("prompt", "") or input.get("text", "")
        else:
            prompt = str(input)
            
        response = self.model.generate_content(prompt)
        # Return just the text string instead of a dictionary
        return response.text
    
    def batch(self, inputs, config=None, **kwargs):
        return [self.invoke(input, config, **kwargs) for input in inputs]
    
    def stream(self, input, config=None, **kwargs):
        raise NotImplementedError("Streaming not implemented for GeminiLLM")
# ------------------ RAG SETUP ------------------
@st.cache_resource
def setup_rag():
    loader = TextLoader("crop_context.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

vector_store = setup_rag()

# ------------------ PROMPT TEMPLATES ------------------
recommendation_template = """
You are an agricultural expert. Analyze these soil and weather conditions:
- Nitrogen (N): {n} ppm
- Phosphorus (P): {p} ppm
- Potassium (K): {k} ppm
- Temperature: {temp}°C
- Humidity: {hum}%
- pH: {ph}
- Rainfall: {rain} mm

Based on the following ideal crop conditions from our database:
{context}

Recommend breafly and in one table the top 3 most suitable crops for these conditions. For each crop:
1. Explain why it's suitable (compare each parameter with ideal ranges)
2. List any minor adjustments needed (specific to each parameter)
3. Provide planting tips (season, spacing, irrigation needs)
4. Mention expected yield potential

Format your response with clear headings for each crop and use markdown for readability.
"""

analysis_template = """
Analyze the feasibility of growing {crop} in these conditions:
- N: {n} ppm (ideal: {ideal_n:.1f})
- P: {p} ppm (ideal: {ideal_p:.1f})
- K: {k} ppm (ideal: {ideal_k:.1f})
- Temp: {temp}°C (ideal: {ideal_temp:.1f})
- Hum: {hum}% (ideal: {ideal_hum:.1f})
- pH: {ph} (ideal: {ideal_ph:.1f})
- Rain: {rain} mm (ideal: {ideal_rain:.1f})

Context from our database:
{context}

Provide short analysis with:
1. brief and short Feasibility score (0-100%) 
2. brief Parameter-by-parameter comparison (tabular format)
3. brief Corrective actions for each mismatched parameter
4. brief Expected yield impact with justification
5. brief Alternative varieties or crops if not feasible
6. brief Risk assessment for each parameter
"""
chat_prompt = PromptTemplate(
    input_variables=["input", "context"],
    template="""You are an agricultural assistant. Use the following context where relevant:
    
    Context: {context}
    
    User question: {input}
    
    Provide a helpful response about farming, crops, or agriculture:"""
)

# Create a chat chain (add this where you create your other chains)


# Create LangChain prompt templates
recommendation_prompt = PromptTemplate(
    input_variables=["n", "p", "k", "temp", "hum", "ph", "rain", "context"],
    template=recommendation_template
)

analysis_prompt = PromptTemplate(
    input_variables=["crop", "n", "p", "k", "temp", "hum", "ph", "rain", 
                    "ideal_n", "ideal_p", "ideal_k", "ideal_temp", 
                    "ideal_hum", "ideal_ph", "ideal_rain", "context"],
    template=analysis_template
)
chat_prompt = PromptTemplate(
    input_variables=["input", "context"],
    template="""
    
    Context: {context}
    
    New question: {input}
    
    Response:"""
)
# Create LLM chains
llm = GeminiLLM()
chat_chain = LLMChain(llm=llm, prompt=chat_prompt)
recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt)
analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)

# ------------------ STREAMLIT UI ------------------

# Sidebar - Location Input
with st.sidebar:
    st.header("📍 Farm Location")
    city = st.text_input("Enter your city", "Sfax")
    crop = st.selectbox("Select your crop", list(watering_guide.keys()), index=2)  # Default to maize
    
    if st.button("Get Weather Data"):
        try:
            weather = get_weather(city)
            st.session_state.weather = weather
            st.success("Weather data loaded!")
            
            # Display basic weather info
            st.subheader("Current Weather")
            st.write(f"🌡️ Temperature: {weather['temperature']}°C")
            st.write(f"💧 Humidity: {weather['humidity']}%")
            st.write(f"🌧️ Rainfall: {weather['rainfall']} mm")
            st.write(f"🌬️ Wind Speed: {weather['wind_speed']} m/s")
            st.write(f"☁️ Condition: {weather['condition']}")
            
            # Get forecast alerts
            forecast_alerts = get_weather_forecast(city)
            if forecast_alerts:
                st.subheader("Weather Alerts")
                for alert in forecast_alerts:
                    st.warning(alert)
            
            # Evaluate conditions
            evaluation = evaluate_conditions(crop, weather)
            st.subheader("Crop Recommendations")
            st.write(f"💧 Recommended water: {evaluation['recommended_water_l_m2']} L/m²")
            st.info(evaluation['advice'])
            
            # Prepare alert message
            message = f"""
                📢 Weather Alert for {city}
                Crop: {evaluation['crop']}
                🌡️ Temp: {evaluation['weather']['temperature']}°C
                🌧️ Rainfall: {evaluation['weather']['rainfall']} mm
                💧 Watering: {evaluation['recommended_water_l_m2']} L/m²
                🛡️ Advice: {evaluation['advice']}
                """
            mail= st.text_input("Enter your email", "ihedhouib@gmail.com")
            whatsapp= st.text_input("Enter your WhatsApp number", "+21626720354")
            if forecast_alerts:
                message += "\n🌦️ Forecast Alerts:\n" + "\n".join(forecast_alerts)
            
            # Send alerts section
            st.subheader("Send Alerts")
            if st.button("Send Email Alert"):
                if send_email(f"Weather Alert for {city}", message,mail):
                    st.success("Email sent successfully!")
                    if send_whatsapp(message,whatsapp):
                        st.success("WhatsApp message sent successfully!")
            
            if st.button("Send WhatsApp Alert"):
                if send_whatsapp(message,whatsapp):
                    st.success("WhatsApp message sent successfully!")
                    if send_email(f"Weather Alert for {city}", message,mail):
                        st.success("Email sent successfully!")
                    
        except Exception as e:
            st.error(f"Failed to fetch weather data: {e}")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["🌱 Dynamic Recommendations", "🔍 Crop Feasibility Analysis", "💬 Chat Assistant"])

with tab1:
    st.subheader("AI-Powered Crop Recommendations")
   
    col1, col2 = st.columns(2)
    with col1:
        n = st.slider("Nitrogen (N) ppm", 0, 150, 50)
        p = st.slider("Phosphorus (P) ppm", 0, 150, 50)
        k = st.slider("Potassium (K) ppm", 0, 150, 50)
    with col2:
        temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
        hum = st.slider("Humidity (%)", 0, 100, 60)
        ph = st.slider("Soil pH", 3.0, 10.0, 6.5)
        rain = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)
   
    if st.button("Generate Recommendations"):
        # Get relevant context from vector store
        query = f"N:{n} P:{p} K:{k} Temp:{temp} Hum:{hum} pH:{ph} Rain:{rain}"
        docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
       
        # Generate recommendations using LangChain
        with st.spinner("Analyzing optimal crops..."):
            response = recommendation_chain.invoke({
                "n": n,
                "p": p,
                "k": k,
                "temp": temp,
                "hum": hum,
                "ph": ph,
                "rain": rain,
                "context": context
            })
            st.session_state.recommendations = response["text"]
       
        st.markdown(st.session_state.recommendations)
        # model=joblib.load('crop_recommendation_model.joblib')
        # probabilities = model.predict_proba(data.iloc[[sample_idx]])[0]

        # col1, col2 = st.columns(2)
        # with col1:
        #     st.info(f"**Actual Crop:** {actual_crop}")

        # with col2:
        #     st.success(f"**Predicted Crop:** {predicted_crop}")

        # Display probabilities
        # st.subheader("Prediction Probabilities")
        # prob_df = pd.DataFrame({
        #     'Crop': model.classes_,
        #     'Probability': probabilities
        # }).sort_values('Probability', ascending=False)

        # st.dataframe(
        #     prob_df.style.background_gradient(cmap='YlGn'),
        #     height=400,
        #     use_container_width=True
        # )

        # # Feature importance
        # st.subheader("Feature Importance")
        # importances = pd.DataFrame({
        #     'Feature': required_features,
        #     'Importance': model.feature_importances_
        # }).sort_values('Importance', ascending=False)

        # st.bar_chart(importances.set_index('Feature'))
        st.header("🌦️ 15-Day Weather Forecast")

        if city:
            try:
                # Get current weather
                current_weather = get_weather(city)

                # Get 15-day forecast
                forecast_data = get_15_day_forecast(city)

                # Evaluate weather impact
                alerts, recommendations = evaluate_weather_impact('rice', forecast_data)

                # Display current weather
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Temperature", f"{current_weather['temperature']}°C")
                col2.metric("Humidity", f"{current_weather['humidity']}%")
                col3.metric("Rainfall", f"{current_weather['rainfall']} mm")
                col4.metric("Condition", current_weather['condition'])

                # Plot forecast
                fig, ax = plt.subplots(figsize=(12, 6))

                # Prepare data for plotting
                dates = [datetime.strptime(day['date'], '%Y-%m-%d') for day in forecast_data]
                avg_temps = [day['avg_temp'] for day in forecast_data]
                rain = [day['total_rain'] for day in forecast_data]

                # Plot temperature
                ax.plot(dates, avg_temps, marker='o', color='red', label='Avg Temperature (°C)')

                # Plot rainfall on secondary axis
                ax2 = ax.twinx()
                ax2.bar(dates, rain, color='blue', alpha=0.3, width=0.5, label='Rainfall (mm)')

                # Formatting
                ax.set_xlabel('Date')
                ax.set_ylabel('Temperature (°C)', color='red')
                ax2.set_ylabel('Rainfall (mm)', color='blue')
                ax.set_title(f'15-Day Weather Forecast for {city}')
                ax.grid(True)
                fig.legend(loc='upper right')

                st.pyplot(fig)

                # Display alerts and recommendations
                if alerts:
                    st.warning("🚨 Weather Alerts")
                    for alert in alerts:
                        st.write(alert)

                    if enable_alerts and (email_receiver or whatsapp_number):
                        alert_message = f"Weather alerts for {city}:\n" + "\n".join(alerts)

                        if st.button("Send Alerts Now"):
                            if email_receiver:
                                send_email(email_receiver, f"Weather Alerts for {city}", alert_message)
                            if whatsapp_number:
                                send_whatsapp(whatsapp_number, alert_message)

                st.subheader("🌱 Crop-Specific Recommendations")
                if recommendations:
                    for rec in recommendations:
                        st.write(f"- {rec}")
                else:
                    st.info("No special recommendations needed for the forecast period.")

            except Exception as e:
                st.error(f"Could not fetch weather data: {str(e)}")

with tab2:
    st.subheader("Crop-Specific Feasibility Analysis")
   
    selected_crop = st.selectbox("Select Crop", data["Crop"].unique())
   
    if st.button("Analyze Feasibility"):
        # Get ideal conditions for selected crop
        ideal = ideal_conditions.loc[selected_crop]
       
        # Get current conditions (from session state or defaults)
        current = {
            'n': st.session_state.get('n', n),
            'p': st.session_state.get('p', p),
            'k': st.session_state.get('k', k),
            'temp': st.session_state.get('temp', temp),
            'hum': st.session_state.get('hum', hum),
            'ph': st.session_state.get('ph', ph),
            'rain': st.session_state.get('rain', rain)
        }
       
        # Get relevant context
        query = f"{selected_crop} ideal conditions"
        docs = vector_store.similarity_search(query, k=2)
        context = "\n".join([doc.page_content for doc in docs])
       
        # Generate analysis using LangChain
        with st.spinner("Evaluating growing conditions..."):
            analysis = analysis_chain.invoke({
                "crop": selected_crop,
                "n": current['n'],
                "p": current['p'],
                "k": current['k'],
                "temp": current['temp'],
                "hum": current['hum'],
                "ph": current['ph'],
                "rain": current['rain'],
                "ideal_n": ideal['Nitrogen'],
                "ideal_p": ideal['Phosphorus'],
                "ideal_k": ideal['Potassium'],
                "ideal_temp": ideal['Temperature'],
                "ideal_hum": ideal['Humidity'],
                "ideal_ph": ideal['pH_Value'],
                "ideal_rain": ideal['Rainfall'],
                "context": context
            })
            st.session_state.analysis = analysis["text"]
       
        st.markdown(st.session_state.analysis)
with tab3:
    st.header("💬 Farming Chat Assistant")
    
    # Initialize chat history
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Clear chat button with better styling
    if st.button("🧹 Clear Chat", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Chat container with scrollable messages
    chat_container = st.container()
    with chat_container:
        # Display messages with avatars
        for message in st.session_state.messages:
            avatar = "👨‍🌾" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    # Accept user input with fixed position
    st.markdown("""
    <style>
        /* Transparent chat input */
        .stChatInput > div > div > input {
            
            background-color: transparent !important;
            border: 1px solid #e1e4e8 !important;
        }
        
        /* Responsive layout */
        @media (max-width: 768px) {
            .stChatInput {
                width: 90% !important;
                left: 5% !important;
            }
        }
        
        /* Message container spacing */
        .stChatMessage {
            margin-bottom: 12px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Fixed position input at bottom
    input_container = st.container()
    with input_container:
        if prompt := st.chat_input("Ask me anything about farming..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message immediately
            with chat_container:
                with st.chat_message("user", avatar="👨‍🌾"):
                    st.markdown(prompt)
            
            # Generate response
            with st.spinner("Thinking..."):
                try:
                    # Get relevant context
                    docs = vector_store.similarity_search(prompt, k=3)
                    context = "\n".join([doc.page_content for doc in docs])
                    
                    # Generate response without history
                    response = chat_chain.run(input=prompt, context=context)
                except Exception as e:
                    response = "Sorry, I encountered an error. Please try again."
            
            # Display and store response
            with chat_container:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Auto-scroll to bottom
            st.markdown(
                "<script>window.scrollTo(0, document.body.scrollHeight);</script>",
                unsafe_allow_html=True
            )
# Weather display (if available)
if 'weather' in st.session_state:
    st.sidebar.subheader("Current Weather")
    st.sidebar.metric("Temperature", f"{st.session_state.weather['temperature']}°C")
    st.sidebar.metric("Humidity", f"{st.session_state.weather['humidity']}%")
    st.sidebar.metric("Rainfall", f"{st.session_state.weather['rainfall']} mm (1h)")