import streamlit as st
import pandas as pd
import re
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os
import tempfile


# Load dataset
df = pd.read_csv("symptom_data.csv")

translator = Translator()
st.set_page_config(page_title="AI Medical Voice Assistant", layout="wide")
st.title("🎙️🩺 AI Medical Voice Assistant")
st.write("Speak or type your symptom in any language (English, Hindi, Telugu...)")

# Initialize variable
speech_text = ""

# Text input (always available)
text_input = st.text_input("Or type your symptom 👇", "")

# Voice input
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
        speech_text = ""  # Prevents recognizer error

# Use text input if no voice input or if voice failed
if not speech_text and text_input:
    speech_text = text_input

# Continue only if we have some symptom text
if speech_text:
    try:
        # Detect language
        detected_lang = translator.detect(speech_text).lang
        st.markdown(f"🌐 Detected Language: `{detected_lang}`")

        # Translate to English
        translated_text = translator.translate(
            speech_text, dest='en').text.lower()
        st.markdown(f"📝 Translated Text: `{translated_text}`")

        # Extract keywords
        keywords = re.findall(r'\b\w+\b', translated_text)

        # Match symptoms
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

            # Translate result
            result_translated = translator.translate(
                result, dest=detected_lang).text
            st.markdown("🌍 **Translated Medical Advice:**")
            st.success(result_translated)

            # Speak result
            tts = gTTS(text=result_translated, lang=detected_lang)
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
