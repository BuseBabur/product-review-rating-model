import joblib
import numpy as np
import re
from complaint_categories_zeroshot import extract_complaints_zeroshot

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

def count_complaints_by_category(texts, threshold=0.5):
    """
    Count complaints by category across multiple texts using zero-shot classification.
    Returns a dictionary with category names as keys and counts as values.
    
    Args:
        texts: List of review texts
        threshold: Confidence threshold for zero-shot (default 0.5)
    
    Returns:
        dict: {category: count} for all categories
    """
    complaint_counts = {
        'material_quality': 0,
        'sound_quality': 0,
        'battery_life': 0,
        'comfort_fit': 0,
        'connectivity': 0,
        'shipping_delivery': 0,
        'price_value': 0,
        'customer_service': 0
    }
    
    for text in texts:
        complaints = extract_complaints_zeroshot(text, threshold=threshold)
        for category in complaints:
            complaint_counts[category] += 1
    
    return complaint_counts

def get_top_complaints_zeroshot(texts, top_n=3, threshold=0.5):
    """
    Get the most common complaints across multiple texts using zero-shot classification.
    Returns a list of tuples (category, count, description).
    
    Args:
        texts: List of review texts
        top_n: Number of top complaints to return
        threshold: Confidence threshold for zero-shot
    
    Returns:
        list: List of tuples (category, count, description)
    """
    complaint_counts = count_complaints_by_category(texts, threshold=threshold)
    
    # Create descriptions for each category
    category_descriptions = {
        'material_quality': 'Issues related to the physical quality and durability of materials',
        'sound_quality': 'Issues related to audio performance and sound characteristics',
        'battery_life': 'Issues related to battery performance and charging',
        'comfort_fit': 'Issues related to physical comfort and fit',
        'connectivity': 'Issues related to wireless connectivity and pairing',
        'shipping_delivery': 'Issues related to shipping, delivery, and packaging',
        'price_value': 'Issues related to pricing and value for money',
        'customer_service': 'Issues related to customer service and support'
    }
    
    # Sort by count and get top N
    sorted_complaints = sorted(
        [(cat, count, category_descriptions[cat]) 
         for cat, count in complaint_counts.items() if count > 0],
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_complaints[:top_n]

def predict_rating_and_complaints(comments):
    """
    Predict ratings and analyze complaints for a list of comments.
    Returns a tuple of (predictions, top_complaints)
    """
    # Clean comments
    clean_comments = [clean_text(c) for c in comments]
    
    # Get predictions
    X = vectorizer.transform(clean_comments)
    if svd is not None:
        X = svd.transform(X)
    else:
        X = X.toarray()
    lexicon_features = extract_lexicon_features(clean_comments)
    meta_features = extract_meta_features(clean_comments)
    X_full = np.hstack([
        X,
        lexicon_features if lexicon_features.ndim == 2 else lexicon_features.reshape(-1, 4),
        meta_features if meta_features.ndim == 2 else meta_features.reshape(-1, 4)
    ])
    predictions = label_encoder.inverse_transform(model.predict(X_full))
    
    # Get top complaints using zero-shot
    top_complaints = get_top_complaints_zeroshot(clean_comments, top_n=3)
    
    return predictions, top_complaints

if __name__ == "__main__":
    comments = [
        "This product is amazing!",
        "Terrible quality, very disappointed.",
        "Average, nothing special.",
        "Fast shipping and good packaging.",
        "Worst purchase ever."
    ]
    predictions, top_complaints = predict_rating_and_complaints(comments)
    
    print("\nPredictions:")
    for comment, rating in zip(comments, predictions):
        print(f"Comment: {comment}\nPredicted Rating: {rating}\n")
    
    print("\nTop Complaints:")
    for category, count, description in top_complaints:
        print(f"{category}: {count} mentions - {description}")
    
    # Test the complaint counting function
    print("\nComplaint Counts by Category:")
    complaint_counts = count_complaints_by_category(comments)
    for category, count in complaint_counts.items():
        if count > 0:
            print(f"{category}: {count}")
