from transformers import pipeline

print("\n===== Emotion Detection System =====\n")

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

text = input("Enter text: ")

result = classifier(text)

emotion = result[0][0]['label']
confidence = result[0][0]['score']

print("\nDetected Emotion:", emotion)
print("Confidence Score:", round(confidence * 100, 2), "%")