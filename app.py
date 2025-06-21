import streamlit as st
import pandas as pd
import re
import speech_recognition as sr
from translate import Translator
from gtts import gTTS
import os
import tempfile

# Load symptom dataset
df = pd.read_csv("symptom_data.csv")

# Streamlit setup
st.set_page_config(page_title="AI Medical Voice Assistant", layout="wide")
st.title("🎙️🩺 AI Medical Voice Assistant")
st.write("Speak or type your symptom in your language (e.g., Hindi, Telugu, etc.)")

# Dropdown to select language
lang_code = st.selectbox("Choose your input language:", [
                         "en", "hi", "te", "ta", "bn", "kn", "ml", "mr", "gu", "ur"], index=1)

# Initialize text
speech_text = ""
text_input = st.text_input("Or type your symptom 👇", "")

# Voice input
if st.button("🎤 Start Listening"):
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎙️ Listening... Speak now.")
            audio = recognizer.listen(source, timeout=5)
            st.success("✅ Audio captured.")
        speech_text = recognizer.recognize_google(audio, language=lang_code)
        st.markdown(f"🗣️ You said: **{speech_text}**")
    except Exception as e:
        st.error(f"🎙️ Microphone error: {e}")
        speech_text = ""

# Use typed input if no voice
if not speech_text and text_input:
    speech_text = text_input

# Proceed if text exists
if speech_text:
    try:
        # Translate to English
        translator = Translator(from_lang=lang_code, to_lang="en")
        translated_text = translator.translate(speech_text).lower()
        st.markdown(f"📝 Translated Text: `{translated_text}`")

        # Extract keywords and match symptoms
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

            # Translate back to user's language
            back_translator = Translator(from_lang="en", to_lang=lang_code)
            result_translated = back_translator.translate(result)
            st.markdown("🌍 **Translated Advice:**")
            st.success(result_translated)

            # Speak the advice
            tts = gTTS(text=result_translated, lang=lang_code)
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
