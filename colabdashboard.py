import gradio as gr
import joblib
import numpy as np
import re
from complaint_categories import get_top_complaints

# Load model and preprocessing objects
try:
    model = joblib.load('best_ensemble_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    try:
        svd = joblib.load('svd.joblib')
    except:
        svd = None
    label_encoder = joblib.load('label_encoder.joblib')
except Exception as e:
    print(f"Error loading model files: {str(e)}")
    raise

# Text cleaning (same as in Untitled-2.py)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'[\W_]+', ' ', text)  # Remove non-word chars
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

def predict_rating_and_complaints(text):
    """
    Predict rating and analyze complaints for a single review.
    Returns both the predicted rating and the top complaints.
    """
    try:
        # Clean and transform the input text
        clean_input = clean_text(text)
        X = vectorizer.transform([clean_input])
        if svd is not None:
            X = svd.transform(X)
        else:
            X = X.toarray()
        
        # Add lexicon and meta features
        lexicon_features = extract_lexicon_features([clean_input])
        meta_features = extract_meta_features([clean_input])
        X_full = np.hstack([X, lexicon_features, meta_features])
        
        # Make prediction
        pred = model.predict(X_full)
        predicted_rating = label_encoder.inverse_transform(pred)[0]
        
        # Get complaints
        top_complaints = get_top_complaints([clean_input], top_n=3)
        
        # Format complaints for display
        complaints_text = "No significant complaints detected."
        if top_complaints:
            complaints_text = "Top Complaints:\n"
            for category, count, description in top_complaints:
                complaints_text += f"- {category}: {description}\n"
        
        return predicted_rating, complaints_text
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return 0, f"Error occurred: {str(e)}"

# Make the function available for import
__all__ = ['predict_rating_and_complaints']

# Only create the interface if this file is run directly
if __name__ == "__main__":
    iface = gr.Interface(
        fn=predict_rating_and_complaints,
        inputs=gr.Textbox(
            label="Enter your review text",
            placeholder="Type your review here...",
            lines=5
        ),
        outputs=[
            gr.Number(label="Predicted Rating (1-5)"),
            gr.Textbox(label="Complaint Analysis", lines=5)
        ],
        title="Review Analysis Dashboard",
        description="Enter a product review to predict its rating and analyze potential complaints.",
        examples=[
            ["The product arrived quickly and fits perfectly. Very satisfied with the quality!"],
            ["Poor quality material, arrived damaged and doesn't fit well. Would not recommend."],
            ["Good product but shipping took longer than expected. The material quality is decent."]
        ]
    )
    iface.launch()
