import streamlit as st
import pandas as pd
st.set_page_config(page_title="AI Powered Crop Recommendation System and Market Advisory", page_icon="🌱")

# ---------------- Load Local CSS ----------------
def local_css(file_name):
    """Load a local CSS file into Streamlit."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ CSS file not found. Using default styling.")

# Apply custom CSS (optional)
local_css("bpp_style.css")

# ---------------- Chatbase Script ----------------
chatbase_script = """
<script>
(function(){if(!window.chatbase||window.chatbase("getState")!=="initialized"){window.chatbase=(...arguments)=>{if(!window.chatbase.q){window.chatbase.q=[]}window.chatbase.q.push(arguments)};window.chatbase=new Proxy(window.chatbase,{get(target,prop){if(prop==="q"){return target.q}return(...args)=>target(prop,...args)}})}const onLoad=function(){const script=document.createElement("script");script.src="https://www.chatbase.co/embed.min.js";script.id="stGSNcflAgIWun5ntpbM5";script.domain="www.chatbase.co";document.body.appendChild(script)};if(document.readyState==="complete"){onLoad()}else{window.addEventListener("load",onLoad)}})();
</script>
"""


st.components.v1.html(chatbase_script, height=700, width=900)


# ---------------- Translations ----------------
translations = {
    "English": {
        "title": "🌱 AGRIMIND AI-POWERED CROP & MARKET ADVISORY By GreenMind",
        "nitrogen": "Enter Nitrogen value",
        "phosphorus": "Enter Phosphorus value",
        "potassium": "Enter Potassium value",
        "ph": "Enter pH value",
        "rainfall": "Enter Rainfall (mm)",
        "temperature": "Enter Temperature (°C)",
        "humidity": "Enter Humidity (%)",
        "button": "🔍 Recommend Crop",
        "success": "Recommended Crop",
        "market": "Market Price",
    },
    "Hindi": {
        "title": "🌱 एआई-संचालित फसल सिफारिश प्रणाली",
        "nitrogen": "नाइट्रोजन का मान दर्ज करें",
        "phosphorus": "फास्फोरस का मान दर्ज करें",
        "potassium": "पोटेशियम का मान दर्ज करें",
        "ph": "pH मान दर्ज करें",
        "rainfall": "वर्षा (मिमी) दर्ज करें",
        "temperature": "तापमान (°C) दर्ज करें",
        "humidity": "आर्द्रता (%) दर्ज करें",
        "button": "🔍 फसल सुझाएं",
        "success": "अनुशंसित फसल",
        "market": "बाजार मूल्य",
    },
}

# ---------------- Load Market Data ----------------
try:
    market_data = pd.read_csv("data/market_prices.csv")
except FileNotFoundError:
    market_data = pd.DataFrame({"crop": [], "price_per_quintal": []})
    st.warning("⚠️ Market price dataset not found. Showing default results.")

# ---------------- Crop Recommendation Function ----------------
def recommend_crop(N, P, K, pH, rainfall, temperature, humidity, market_data):
    if 6 <= pH <= 7 and rainfall > 100 and humidity > 70:
        crop = "Rice"
    elif 5.5 <= pH <= 6.5 and N < 100 and temperature < 25:
        crop = "Wheat"
    else:
        crop = "Maize"

    price_row = market_data[market_data["crop"].str.lower() == crop.lower()]
    if not price_row.empty:
        price = price_row["price_per_quintal"].values[0]
    else:
        price = "N/A"

    return {"recommended_crop": crop, "market_price": price}

# ---------------- Streamlit UI ----------------
lang = st.sidebar.radio("🌐 Select Language / भाषा चुनें", ["English", "Hindi"])
st.title(translations[lang]["title"])

N = st.number_input(translations[lang]["nitrogen"], min_value=0, max_value=140, value=50)
P = st.number_input(translations[lang]["phosphorus"], min_value=0, max_value=140, value=50)
K = st.number_input(translations[lang]["potassium"], min_value=0, max_value=140, value=50)
pH = st.number_input(translations[lang]["ph"], min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input(translations[lang]["rainfall"], min_value=0, max_value=500, value=100)
temperature = st.number_input(translations[lang]["temperature"], min_value=-10, max_value=50, value=25)
humidity = st.number_input(translations[lang]["humidity"], min_value=0, max_value=100, value=70)

if st.button(translations[lang]["button"]):
    result = recommend_crop(N, P, K, pH, rainfall, temperature, humidity, market_data)
    st.success(
        f"{translations[lang]['success']}: {result['recommended_crop']} | "
        f"{translations[lang]['market']}: ₹{result['market_price']}/quintal"
    )

# ---------------- Floating External Link ----------------
st.markdown(
    """
    <div style="position: fixed; bottom: 60px; right: 10px; background-color: #f0f0f0;
                padding: 8px; border-radius: 5px;">
        <a href="https://ai-powered-leaf-stress-detection-for-sustainable-farming-ngsvn.streamlit.app/"
           target="_blank"
           style="text-decoration: none; color: blue; font-weight: bold;">
            🌱 Check Leaf Health
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- Inject Chatbase Chatbot (floating bottom-right) ----------------
st.markdown(chatbase_script, unsafe_allow_html=True)
