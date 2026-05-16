from flask import Flask, render_template, request, jsonify
from langdetect import detect
from deep_translator import GoogleTranslator
import requests
import os

app = Flask(__name__)

# ── Hugging Face config ────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = (
    "https://api-inference.huggingface.co/models/"
    "j-hartmann/emotion-english-distilroberta-base"
)

# ── Lookup tables ──────────────────────────────────────────────────────────────
LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "fr": "French", "es": "Spanish",
    "de": "German", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "zh-cn": "Chinese", "ar": "Arabic", "ru": "Russian",
}
EMOTION_EMOJIS = {
    "joy": "😊", "sadness": "😢", "anger": "😡", "fear": "😨",
    "surprise": "😲", "disgust": "🤢", "neutral": "😐",
}
EMOTION_COLORS = {
    "joy": "text-green-400", "sadness": "text-blue-400", "anger": "text-red-400",
    "fear": "text-yellow-400", "surprise": "text-purple-400",
    "disgust": "text-pink-400", "neutral": "text-indigo-300",
}


# ── Health check — required by render.yaml ─────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# ── Emotion detection ──────────────────────────────────────────────────────────
def detect_emotion(text: str) -> dict:
    if not HF_TOKEN:
        return {"error": "HF_TOKEN environment variable is not set."}

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        response = requests.post(
            API_URL, headers=headers, json={"inputs": text}, timeout=20,
        )
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.Timeout:
        return {"error": "The AI model is loading — please retry in a few seconds."}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}
    except ValueError:
        return {"error": "Invalid JSON returned by the API."}

    if isinstance(result, dict) and "error" in result:
        return {"error": result["error"]}
    if not result:
        return {"error": "Empty response from the API."}

    try:
        candidates = result[0] if isinstance(result[0], list) else result
        return max(candidates, key=lambda x: x["score"])
    except (KeyError, IndexError, TypeError) as e:
        return {"error": f"Unexpected API response format: {e}"}


# ── Main route ─────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        text = request.form.get("text", "").strip()

        if not text:
            result = {"error": "Please enter some text."}
            return render_template("index.html", result=result)

        try:
            lang_code = detect(text)
            language = LANGUAGE_NAMES.get(lang_code, f"Unknown ({lang_code})")

            translated_text = (
                text if lang_code == "en"
                else GoogleTranslator(source="auto", target="en").translate(text)
            )

            prediction = detect_emotion(translated_text)

            if "error" in prediction:
                result = {"error": prediction["error"]}
                return render_template("index.html", result=result)

            emotion = prediction["label"].lower()
            result = {
                "language":      language,
                "translated":    translated_text,
                "emotion":       emotion.capitalize(),
                "confidence":    round(prediction["score"] * 100, 2),
                "emoji":         EMOTION_EMOJIS.get(emotion, "🧠"),
                "emotion_class": EMOTION_COLORS.get(emotion, "text-indigo-300"),
                "error":         None,
            }
        except Exception as e:
            result = {"error": str(e)}

    return render_template("index.html", result=result)


# ── Entry point (local dev only — Render uses Gunicorn) ────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)