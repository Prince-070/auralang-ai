from flask import Flask, render_template, request

from langdetect import detect

import requests
import os

from deep_translator import GoogleTranslator

app = Flask(**name**)

print("Loading AI Emotion Model...")

# Hugging Face API Token

HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face Emotion Model API

API_URL = (
"https://api-inference.huggingface.co/models/"
"j-hartmann/emotion-english-distilroberta-base"
)

headers = {
"Authorization": f"Bearer {HF_TOKEN}"
}

print("AI Model Loaded Successfully!")

# Emotion Detection Function

def detect_emotion(text):


payload = {
    "inputs": text
}

response = requests.post(
    API_URL,
    headers=headers,
    json=payload
)

try:

    result = response.json()

    return result

except Exception:

    return {
        "error": "Unable to process emotion detection."
    }


# Language Mapping

language_names = {


"en": "English",
"hi": "Hindi",
"fr": "French",
"es": "Spanish",
"de": "German",
"it": "Italian",
"ja": "Japanese",
"ko": "Korean",
"zh-cn": "Chinese",
"ar": "Arabic",
"ru": "Russian"


}

# Emotion Emojis

emotion_emojis = {


"joy": "😊",
"sadness": "😢",
"anger": "😡",
"fear": "😨",
"surprise": "😲",
"disgust": "🤢",
"neutral": "😐"


}

# Emotion Colors

emotion_colors = {


"joy": "text-green-400",
"sadness": "text-blue-400",
"anger": "text-red-400",
"fear": "text-yellow-400",
"surprise": "text-purple-400",
"disgust": "text-pink-400",
"neutral": "text-indigo-300"


}
@app.route("/", methods=["GET", "POST"])
def home():


result = None

if request.method == "POST":

    text = request.form["text"]

    try:

        # Detect Language
        lang_code = detect(text)

        language = language_names.get(
            lang_code,
            "Unknown"
        )

        # Translate To English
        translated_text = GoogleTranslator(
            source='auto',
            target='en'
        ).translate(text)

        # Emotion Detection
        prediction = detect_emotion(
            translated_text
        )

        print(prediction)

        # Handle API Errors
        if isinstance(prediction, dict):

            if "error" in prediction:

                result = {
                    "error": prediction["error"]
                }

                return render_template(
                    "index.html",
                    result=result
                )

        # Extract Emotion
        emotion = prediction[0]["label"]

        confidence = round(
            prediction[0]["score"] * 100,
            2
        )

        # Emotion Emoji
        emoji = emotion_emojis.get(
            emotion.lower(),
            "🧠"
        )

        # Emotion Color
        emotion_class = emotion_colors.get(
            emotion.lower(),
            "text-indigo-300"
        )

        # Final Result
        result = {

            "language": language,

            "translated": translated_text,

            "emotion": emotion.capitalize(),

            "confidence": confidence,

            "emoji": emoji,

            "emotion_class": emotion_class,

            "error": None

        }

    except Exception as e:

        result = {

            "error": str(e)

        }

return render_template(
    "index.html",
    result=result
)


if **name** == "**main**":


app.run(
    host="0.0.0.0",
    port=5000
)

