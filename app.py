import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Powered Crop Recommendation System and Market Advisory", page_icon="🌱")

# Load dataset
market_data = pd.read_csv("data/market_prices.csv")

# 🌐 Language selector
lang = st.selectbox("🌐 Select Language / भाषा चुनें", ["English", "हिन्दी"])

# 🌐 Translation dictionary
translations = {
    "English": {
        "title": "🌱 AGRIMIND AI-POWERED CROP & MARKET ADVISORY By GreenMind",
        "desc": "Enter your soil and weather values to get the best crop recommendation.",
        "inputs": {
            "N": "Nitrogen (N)",
            "P": "Phosphorus (P)",
            "K": "Potassium (K)",
            "pH": "Soil pH",
            "rainfall": "Rainfall (mm)",
            "temperature": "Temperature (°C)",
            "humidity": "Humidity (%)"
        },
        "button": "🔍 Recommend Crop",
        "success": "✅ Recommended Crop",
        "error": "⚠️ Soil type not found in dataset. Please type correctly.",
        "market": "💰 Market Price"
    },
    "हिन्दी": {
        "title": "🌱 एग्रीमाइंड एआई-आधारित फसल और बाज़ार सलाह - ग्रीनमाइंड द्वारा",
        "desc": "अपनी मिट्टी और मौसम का मान दर्ज करें और सर्वश्रेष्ठ फसल की सिफारिश प्राप्त करें।",
        "inputs": {
            "N": "नाइट्रोजन (N)",
            "P": "फास्फोरस (P)",
            "K": "पोटैशियम (K)",
            "pH": "मिट्टी का pH",
            "rainfall": "वर्षा (मिमी)",
            "temperature": "तापमान (°C)",
            "humidity": "आर्द्रता (%)"
        },
        "button": "🔍 फसल की सिफारिश करें",
        "success": "✅ अनुशंसित फसल",
        "error": "⚠️ मिट्टी का प्रकार डेटा में नहीं मिला। कृपया सही लिखें।",
        "market": "💰 बाज़ार मूल्य"
    }
}

# 🏷️ Title & description
st.title(translations[lang]["title"])
st.write(translations[lang]["desc"])

# 📥 User Inputs
N = st.number_input(translations[lang]["inputs"]["N"], min_value=0, max_value=500, value=50)
P = st.number_input(translations[lang]["inputs"]["P"], min_value=0, max_value=500, value=30)
K = st.number_input(translations[lang]["inputs"]["K"], min_value=0, max_value=500, value=40)
pH = st.number_input(translations[lang]["inputs"]["pH"], min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input(translations[lang]["inputs"]["rainfall"], min_value=0.0, max_value=500.0, value=120.0)
temperature = st.number_input(translations[lang]["inputs"]["temperature"], min_value=8, max_value=43, value=25)
humidity = st.number_input(translations[lang]["inputs"]["humidity"], min_value=14, max_value=100, value=80)

# 🌍 External link (bottom-right corner)
st.markdown(
    """
    <div style="position: fixed; bottom: 10px; right: 10px; background-color: #f0f0f0; padding: 8px; border-radius: 5px;">
        <a href="https://ai-powered-leaf-stress-detection-for-sustainable-farming-ngsvn.streamlit.app/" target="_blank" style="text-decoration: none; color: blue; font-weight: bold;">
            🌱 Check Leaf Health
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# 🔮 Crop recommendation logic
def recommend_crop(N, P, K, pH, rainfall, market_data):
    if 6 <= pH <= 7 and rainfall > 100:
        crop = "Rice"
    elif 5.5 <= pH <= 6.5 and N < 100:
        crop = "Wheat"
    else:
        crop = "Maize"
    
    # Find market price
    price_row = market_data[market_data["crop"] == crop]
    if not price_row.empty:
        price = price_row["price_per_quintal"].values[0]
    else:
        price = "N/A"
    
    return {
        "recommended_crop": crop,
        "market_price": price
    }

# 🔘 Button action
if st.button(translations[lang]["button"]):
    result = recommend_crop(N, P, K, pH, rainfall, market_data)
    st.success(
        f"{translations[lang]['success']}: {result['recommended_crop']} | {translations[lang]['market']}: ₹{result['market_price']}/quintal"
    )
