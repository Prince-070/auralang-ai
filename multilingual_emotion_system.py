from langdetect import detect
from transformers import pipeline
from deep_translator import GoogleTranslator

print("\n===== Multilingual Emotion Detection System =====\n")

# Load emotion detection model
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

# Language dictionary
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

# User input
text = input("Enter text: ")

try:
    # Detect language
    lang_code = detect(text)

    language = language_names.get(lang_code, "Unknown")

    print("\nDetected Language:", language)

    # Translate text into English
    translated_text = GoogleTranslator(
        source='auto',
        target='en'
    ).translate(text)

    print("Translated Text:", translated_text)

    # Emotion Detection
    result = classifier(translated_text)

    emotion = result[0][0]['label']
    confidence = result[0][0]['score']

    print("\nDetected Emotion:", emotion)
    print("Confidence Score:", round(confidence * 100, 2), "%")

except Exception as e:
    print("\nError:", e)