from transformers import pipeline

# Load pre-trained sentiment model (downloads once ~250MB)
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Predict on any sentence
result = classifier("This movie was absolutely brilliant!")
print(result)
sentences = [
    "This movie was not bad at all",      # tricky: double negative
    "I can't say I didn't enjoy it",      # tricky: negation
    "The acting was okay I guess",         # ambiguous
    "Absolutely terrible, waste of time",  # clearly negative
    "One of the best films ever made"       # clearly positive
]

results = classifier(sentences)

for sentence, result in zip(sentences, results):
    label = result['label']
    score = round(result['score'] * 100, 1)
    print(f"{label} ({score}%) — {sentence}")