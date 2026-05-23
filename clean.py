import re
import pandas as pd

stop_words = set([
    'i','me','my','we','our','you','your','he','she','it','they',
    'is','are','was','were','be','been','being','have','has','had',
    'do','does','did','a','an','the','and','or','but','in','on',
    'at','to','for','of','with','this','that','these','those','so',
    'if','as','by','from','not','no','nor','can','will','just','more'
])

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df = pd.read_csv('IMDB dataset.csv')
df['clean_review'] = df['review'].apply(clean_text)

# This line adds the label column
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

df.to_csv('cleaned_reviews.csv', index=False)
print("Done!")
print(df.columns.tolist())  # confirms all columns saved