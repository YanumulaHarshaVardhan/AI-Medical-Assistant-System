"""
RAG Medical Assistant — Streamlit App (starter) — Enhanced with Voice + Translation
-------------------------------------------------------------------------------
This version merges the multilingual + voice input features into the RAG skeleton.
- Preserves CLI + Streamlit modes with safe fallbacks.
- Adds optional speech recognition, translation, and TTS when libraries are installed.
- Retains test suite for matching logic.

Usage:
  python app.py --cli --query "I have a fever"
  python app.py --run-tests
  streamlit run app.py   (if streamlit installed)
"""

import os
import sys
import csv
import re
import argparse
import difflib
import unittest
from typing import List, Dict, Optional, Tuple

# ---------- Optional external features
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    st = None
    STREAMLIT_AVAILABLE = False

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from translate import Translator
except Exception:
    Translator = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        return None


# ---------- UI wrappers
def ui_info(msg: str):
    if STREAMLIT_AVAILABLE:
        st.info(msg)
    else:
        print("[INFO]", msg)


def ui_warning(msg: str):
    if STREAMLIT_AVAILABLE:
        st.warning(msg)
    else:
        print("[WARN]", msg)


def ui_error(msg: str):
    if STREAMLIT_AVAILABLE:
        st.error(msg)
    else:
        print("[ERROR]", msg)


def ui_success(msg: str):
    if STREAMLIT_AVAILABLE:
        st.success(msg)
    else:
        print("[OK]", msg)


# ---------- Core matching logic
CSV_FIELDS = ["symptom", "conditions",
              "medicines", "eat", "avoid", "doctor_advice"]


def read_symptom_csv(path: str = "symptom_data.csv") -> List[Dict[str, str]]:
    if not os.path.exists(path):
        ui_warning(f"Symptom CSV not found at: {path}. Returning empty list.")
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            normalized = {k: (r.get(k, "").strip() if r.get(k) else "")
                          for k in CSV_FIELDS}
            normalized["symptom_norm"] = normalize_text(
                normalized.get("symptom", ""))
            rows.append(normalized)
    return rows


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def find_best_match(user_text: str, rows: List[Dict[str, str]], min_score: float = 0.15) -> Tuple[Optional[Dict[str, str]], float]:
    if not user_text or not rows:
        return None, 0.0
    ut = normalize_text(user_text)
    best, best_score = None, 0.0
    ut_tokens = set(ut.split())
    for r in rows:
        symptom = r.get("symptom_norm", "")
        if not symptom:
            continue
        symptom_tokens = set(symptom.split())
        if symptom_tokens & ut_tokens:
            score = len(symptom_tokens & ut_tokens) / \
                max(1, len(symptom_tokens))
            if symptom in ut:
                score = max(score, 0.9)
        else:
            score = max(jaccard_similarity(ut, symptom), difflib.SequenceMatcher(
                None, ut, symptom).ratio() * 0.8)
        if score > best_score:
            best, best_score = r, score
    return (best, best_score) if best_score >= min_score else (None, best_score)


def format_result_row(row: Dict[str, str]) -> str:
    if not row:
        return "No matching symptom found."
    return (f"Symptom: {row.get('symptom', '')}\n"
            f"Possible Conditions: {row.get('conditions', '')}\n"
            f"Medicines: {row.get('medicines', '')}\n"
            f"Recommended Food: {row.get('eat', '')}\n"
            f"Avoid: {row.get('avoid', '')}\n"
            f"See Doctor If: {row.get('doctor_advice', '')}")


# ---------- Optional voice + translation pipeline
def process_with_translation(text: str, from_lang: str = "en", to_lang: str = "en") -> str:
    if Translator is None:
        return text
    try:
        translator = Translator(from_lang=from_lang, to_lang=to_lang)
        return translator.translate(text)
    except Exception:
        return text


def speech_to_text(lang_code="en") -> Optional[str]:
    if sr is None:
        return None
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            ui_info("Listening...")
            audio = recognizer.listen(source, timeout=5)
        return recognizer.recognize_google(audio, language=lang_code)
    except Exception:
        return None


def text_to_speech(text: str, lang_code="en", out_path="output.mp3"):
    if gTTS is None:
        return None
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save(out_path)
        return out_path
    except Exception:
        return None


# ---------- Streamlit UI
def run_streamlit_mode(csv_path="symptom_data.csv"):
    if not STREAMLIT_AVAILABLE:
        ui_error("Streamlit not available.")
        return
    load_dotenv()
    st.set_page_config(page_title="RAG Medical Assistant", layout="wide")
    st.title("🩺 RAG Medical Assistant — with Voice + Translation")
    rows = read_symptom_csv(csv_path)
    lang_code = st.selectbox("Choose input language:", [
                             "en", "hi", "te", "ta", "bn", "kn", "ml", "mr", "gu", "ur"], index=0)
    input_text = st.text_input("Describe your symptom")
    if st.button("🎤 Speak") and sr:
        spoken = speech_to_text(lang_code)
        if spoken:
            input_text = spoken
            st.write(f"🗣️ You said: {input_text}")
    if input_text:
        translated = process_with_translation(
            input_text, from_lang=lang_code, to_lang="en")
        match, score = find_best_match(translated, rows)
        if match:
            result = format_result_row(match)
            st.text_area("Medical Advice", value=result, height=220)
            back_translated = process_with_translation(
                result, from_lang="en", to_lang=lang_code)
            st.success(back_translated)
            mp3_path = text_to_speech(back_translated, lang_code)
            if mp3_path:
                with open(mp3_path, 'rb') as audio_file:
                    st.audio(audio_file.read(), format='audio/mp3')
                os.remove(mp3_path)
        else:
            st.warning("No confident match found.")


# ---------- CLI fallback mode
def run_cli_mode(csv_path="symptom_data.csv", query: Optional[str] = None):
    rows = read_symptom_csv(csv_path)
    if not rows:
        print("No symptom data loaded.")
        return
    if query:
        match, score = find_best_match(query, rows)
        print(format_result_row(match))
    else:
        if not sys.stdin.isatty():
            print("Interactive input not available. Demo query: 'fever'")
            match, score = find_best_match("fever", rows)
            print(format_result_row(match))
            return
        while True:
            try:
                text = input("Describe your symptom: ").strip()
            except (EOFError, KeyboardInterrupt, OSError):
                print("\nExiting.")
                break
            if text.lower() in ("exit", "quit"):
                break
            match, score = find_best_match(text, rows)
            print(format_result_row(match))


# ---------- Tests
class TestSymptomMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [
            {"symptom": "headache", "symptom_norm": "headache", "conditions": "Tension headache",
                "medicines": "Paracetamol", "eat": "Water", "avoid": "Caffeine", "doctor_advice": "If lasts more than 2 days"},
            {"symptom": "stomach pain", "symptom_norm": "stomach pain", "conditions": "Indigestion",
                "medicines": "Antacids", "eat": "Rice", "avoid": "Spicy food", "doctor_advice": "If severe"},
            {"symptom": "fever", "symptom_norm": "fever", "conditions": "Viral infection",
                "medicines": "Paracetamol", "eat": "Coconut water", "avoid": "Fried food", "doctor_advice": "If >102F"}
        ]

    def test_exact_match(self):
        row, score = find_best_match("I have a headache", self.rows)
        self.assertEqual(row["symptom"], "headache")

    def test_partial_match(self):
        row, score = find_best_match("stomach hurts", self.rows)
        self.assertEqual(row["symptom"], "stomach pain")

    def test_no_match(self):
        row, score = find_best_match("broken arm", self.rows)
        self.assertIsNone(row)

    def test_low_score(self):
        row, score = find_best_match("zzzz", self.rows)
        self.assertIsNone(row)


# ---------- Entrypoint
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true")
    parser.add_argument("--csv", type=str, default="symptom_data.csv")
    parser.add_argument("--query", type=str, help="Symptom query for CLI mode")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    if args.run_tests:
        unittest.main(argv=[sys.argv[0]])
        return
    if STREAMLIT_AVAILABLE and not args.cli:
        run_streamlit_mode(args.csv)
    else:
        run_cli_mode(args.csv, query=args.query)


if __name__ == "__main__":
    main()
