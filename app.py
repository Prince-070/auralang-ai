from flask import Flask, render_template, request

from langdetect import detect

from transformers import pipeline

from deep_translator import GoogleTranslator


app = Flask(__name__)


print("Loading AI Emotion Model...")


# Load Hugging Face emotion model
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)


print("AI Model Loaded Successfully!")


# Language mapping
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


# Emotion emojis
emotion_emojis = {

    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐"

}


# Emotion color classes
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

            # Translate text to English
            translated_text = GoogleTranslator(
                source='auto',
                target='en'
            ).translate(text)

            # Emotion Detection
            prediction = classifier(translated_text)

            emotion = prediction[0][0]['label']

            confidence = round(
                prediction[0][0]['score'] * 100,
                2
            )

            # Emotion emoji
            emoji = emotion_emojis.get(
                emotion.lower(),
                "🧠"
            )

            # Emotion color
            emotion_class = emotion_colors.get(
                emotion.lower(),
                "text-indigo-300"
            )

            # Final result dictionary
            result = {

                "language": language,

                "translated": translated_text,

                "emotion": emotion.capitalize(),

                "confidence": confidence,

                "emoji": emoji,

                "emotion_class": emotion_class

            }

        except Exception as e:

            result = {

                "error": str(e)

            }

    return render_template(
        "index.html",
        result=result
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)