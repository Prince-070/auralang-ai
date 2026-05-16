from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException

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
    "ru": "Russian",
    "so": "Somali"
}

print("\n===== Language Detection System =====\n")

text = input("Enter text: ")

try:
    if len(text.strip()) < 5:
        print("\nPlease enter longer text.")
    else:
        predictions = detect_langs(text)

        top_prediction = predictions[0]

        lang_code = top_prediction.lang
        confidence = round(top_prediction.prob * 100, 2)

        language = language_names.get(lang_code, "Unknown")

        print("\nDetected Language Code:", lang_code)
        print("Detected Language Name:", language)
        print("Confidence:", confidence, "%")

except LangDetectException:
    print("\nError: Could not detect language.")