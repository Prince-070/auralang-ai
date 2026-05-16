from flask import Flask, render_template, request, jsonify
from langdetect import detect
from deep_translator import GoogleTranslator
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

# ── Hugging Face config ────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")

# New router URL (old api-inference.huggingface.co is deprecated/404)
# huggingface_hub InferenceClient handles auth + correct endpoint automatically
MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

# ── Lookup tables ──────────────────────────────────────────────────────────────
LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "fr": "French", "es": "Spanish",
    "de": "German",  "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "zh-cn": "Chinese", "ar": "Arabic", "ru": "Russian",
}
EMOTION_EMOJIS = {
    "joy": "😊", "sadness": "😢", "anger": "😡", "fear": "😨",
    "surprise": "😲", "disgust": "🤢", "neutral": "😐",
}
EMOTION_COLORS = {
    "joy":      "text-green-400",
    "sadness":  "text-blue-400",
    "anger":    "text-red-400",
    "fear":     "text-yellow-400",
    "surprise": "text-purple-400",
    "disgust":  "text-pink-400",
    "neutral":  "text-indigo-300",
}


# ── Health check — required by render.yaml ─────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# ── Emotion detection ──────────────────────────────────────────────────────────
def detect_emotion(text: str) -> dict:
    """
    Use huggingface_hub InferenceClient (provider='hf-inference') which calls:
        https://router.huggingface.co/hf-inference/models/<model>
    This replaces the deprecated api-inference.huggingface.co endpoint.

    Returns {"label": "joy", "score": 0.97} or {"error": "..."}
    """
    if not HF_TOKEN:
        return {"error": "HF_TOKEN environment variable is not set."}

    try:
        client = InferenceClient(
            provider="hf-inference",
            api_key=HF_TOKEN,
        )
        # text_classification returns a list of ClassificationOutput objects
        results = client.text_classification(text, model=MODEL_ID)

        if not results:
            return {"error": "Empty response from the API."}

        # Pick highest-scoring label
        best = max(results, key=lambda x: x.score)
        return {"label": best.label, "score": best.score}

    except Exception as e:
        err = str(e)
        # Friendly message for model cold-start (common on free HF tier)
        if "loading" in err.lower() or "503" in err:
            return {"error": "Model is warming up — please retry in a few seconds."}
        return {"error": f"Inference failed: {err}"}


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
            # 1. Detect language
            lang_code = detect(text)
            language = LANGUAGE_NAMES.get(lang_code, f"Unknown ({lang_code})")

            # 2. Translate to English (skip if already English)
            if lang_code == "en":
                translated_text = text
            else:
                translated_text = GoogleTranslator(
                    source="auto", target="en"
                ).translate(text)

            # 3. Detect emotion
            prediction = detect_emotion(translated_text)

            # 4. Surface any errors
            if "error" in prediction:
                result = {"error": prediction["error"]}
                return render_template("index.html", result=result)

            # 5. Build result dict
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