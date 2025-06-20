import pandas as pd
import joblib
import numpy as np
import re

# Load model and preprocessing objects
model = joblib.load('best_ensemble_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
try:
    svd = joblib.load('svd.joblib')
except:
    svd = None
label_encoder = joblib.load('label_encoder.joblib')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[\W_]+', ' ', text)
    text = text.lower()
    return text

def extract_lexicon_features(texts):
    positive_words = {
        'accessible', 'advantageous', 'affordable', 'authentic', 'awesome', 'balanced',
        'beautiful', 'best', 'brilliant', 'clean', 'comfy', 'comfortable', 'consistent',
        'convenient', 'cool', 'cute', 'delighted', 'durable', 'efficient', 'enjoyable',
        'exceptional', 'excellent', 'fantastic', 'fast', 'favorite', 'fit', 'flawless',
        'fresh', 'friendly', 'genuine', 'good', 'grateful', 'happy', 'helpful', 'ideal',
        'impressed', 'impressive', 'liked', 'love', 'loved', 'organized', 'outstanding',
        'perfect', 'pleased', 'premium', 'professional', 'prompt', 'pure', 'quality',
        'quick', 'reasonable', 'recommend', 'recommended', 'reliable', 'responsive',
        'same','satisfied', 'seamless', 'smart', 'smooth', 'sturdy', 'tasty',
        'trustworthy','valuable', 'well', 'wonderful', 'worth', 'worthy'
    }
    negative_words = {
        'annoyed', 'annoying', 'avoid', 'bitter', 'broken', 'bug',
        'comfy_NEG', 'comfortable_NEG', 'confused', 'costly', 'cracks', 'cracked',
        'crap', 'crappy', 'damaged', 'defective', 'delayed', 'deteriorated', 'dirty',
        'disappointed', 'disappointment', 'dishonest', 'disgusting', 'dislike',
        'dissatisfied', 'expired', 'failed', 'fake', 'faking', 'faulty', 'flaw',
        'flaws', 'flimsy', 'fraudulent', 'frustrate', 'frustrating', 'good_NEG',
        'greasy', 'gross', 'harmful', 'hate', 'hated', 'hating', 'helpful_NEG',
        'horrible', 'ignored', 'incompetent', 'incomplete',
        'inconsistent', 'inferior', 'inappropriate', 'lag', 'lagged', 'lagging',
        'leaking', 'liar', 'lies', 'lie', 'low', 'malfunctioning', 'misguide',
        'misguided', 'mishandled', 'mislead', 'misleading', 'moldy', 'moth',
        'neglected', 'overpriced', 'poor', 'pricey', 'problem', 'recommend_NEG',
        'respond_NEG','returned', 'ridiculous', 'rot', 'rotten', 'rude', 'same_NEG',
        'scam', 'scammed', 'shit', 'shitty','spoiled', 'stinks', 'stinky', 'stupid',
        'suspicious', 'terrible', 'toxic','slow','uncomfortable', 'uncomfy', 'unhelpful',
        'unreliable','unresponsive','upset', 'useless', 'waste', 'worst', 'wrong'
    }
    negative_words.update({f"{word}_NEG" for word in positive_words})
    intensifiers = {'very', 'really', 'extremely', 'absolutely', 'totally', 'completely','lot','lots','definitelly',
                    'much','many','freaking','overwhelmingly','especially','quite','seriously','truly'}
    features = []
    for text in texts:
        words = text.split() if isinstance(text, str) else []
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        int_count = sum(1 for w in words if w in intensifiers)
        sentiment_score = pos_count - neg_count
        features.append([pos_count, neg_count, int_count, sentiment_score])
    return np.array(features)

def extract_meta_features(texts):
    features = []
    for text in texts:
        if not isinstance(text, str):
            features.append([0, 0, 0, 0])
            continue
        words = text.split()
        review_length = len(text)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        punct_count = sum(1 for c in text if c in '.,!?')
        upper_count = sum(1 for w in words if w.isupper())
        features.append([review_length, avg_word_length, punct_count, upper_count])
    return np.array(features)

# Load your CSV
df = pd.read_csv('trendyol_reviews_89234146.csv')  # <-- Change filename if needed
df['clean_text'] = df['Text'].apply(clean_text)

# Feature extraction
X = vectorizer.transform(df['clean_text'])
if svd is not None:
    X = svd.transform(X)
else:
    X = X.toarray()
lexicon_features = extract_lexicon_features(df['clean_text'])
meta_features = extract_meta_features(df['clean_text'])
X_full = np.hstack([
    X,
    lexicon_features if lexicon_features.ndim == 2 else lexicon_features.reshape(-1, 4),
    meta_features if meta_features.ndim == 2 else meta_features.reshape(-1, 4)
])

# Predict
pred = model.predict(X_full)
df['Predicted_Score'] = label_encoder.inverse_transform(pred)

# Save results
df.to_csv('trendyol_reviews_89234146_with_predictions.csv', index=False)
print("Predictions saved to trendyol_reviews_782213857_with_predictions.csv")

# Print actual and predicted scores, and count correct/wrong predictions
correct = 0
wrong = 0

print("Actual vs Predicted Scores:")
for actual, predicted in zip(df['Score'], df['Predicted_Score']):
    print(f"Actual: {actual} | Predicted: {predicted}")
    if actual == predicted:
        correct += 1
    else:
        wrong += 1

total = correct + wrong
accuracy = correct / total if total > 0 else 0

print(f"\nTotal reviews: {total}")
print(f"Correct predictions: {correct}")
print(f"Wrong predictions: {wrong}")
print(f"Accuracy: {accuracy:.4f}")