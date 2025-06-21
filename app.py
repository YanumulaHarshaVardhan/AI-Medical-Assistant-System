import streamlit as st
import pandas as pd
import re
import speech_recognition as sr
from translate import Translator
from gtts import gTTS
import os
import tempfile

# Load dataset
df = pd.read_csv("symptom_data.csv")

# Initialize Translator (auto-detect is not available, so default to 'en')
translator = Translator(to_lang="en")

st.set_page_config(page_title="AI Medical Voice Assistant", layout="wide")
st.title("🎙️🩺 AI Medical Voice Assistant")
st.write("Speak or type your symptom in your language (English, Hindi, Telugu...)")

# Input setup
speech_text = ""
text_input = st.text_input("Or type your symptom 👇", "")
use_voice = st.button("🎤 Start Listening")

if use_voice:
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎙️ Listening... Please speak now.")
            audio = recognizer.listen(source, timeout=5)
            st.success("✅ Audio captured. Processing...")
        speech_text = recognizer.recognize_google(audio)
        st.markdown(f"🗣️ You said: **{speech_text}**")
    except Exception as e:
        st.error(f"🎙️ Microphone error: {e}")
        speech_text = ""

if not speech_text and text_input:
    speech_text = text_input

if speech_text:
    try:
        translated_text = translator.translate(speech_text).lower()
        st.markdown(f"📝 Translated Text: `{translated_text}`")

        keywords = re.findall(r'\b\w+\b', translated_text)
        matches = df[df['symptom'].apply(
            lambda x: any(kw in x.lower() for kw in keywords))]

        if not matches.empty:
            row = matches.iloc[0]
            result = f"🩺 Symptom: {row['symptom']}\n" \
                f"🧾 Possible Conditions: {row['conditions']}\n" \
                f"💊 Medicines: {row['medicines']}\n" \
                f"🥗 Recommended Food: {row['eat']}\n" \
                f"❌ Avoid: {row['avoid']}\n" \
                f"👨‍⚕️ See Doctor If: {row['doctor_advice']}"
            st.text_area("💡 Medical Advice", value=result, height=200)

            # Voice Output
            tts = gTTS(text=result, lang="en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_path = fp.name
                tts.save(temp_path)
            with open(temp_path, 'rb') as audio_file:
                st.audio(audio_file.read(), format='audio/mp3')
            os.remove(temp_path)
        else:
            st.warning("❌ No matching symptom found.")
    except Exception as e:
        st.error(f"❌ Error while processing: {e}")
