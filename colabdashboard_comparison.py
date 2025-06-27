import gradio as gr
import joblib
import numpy as np
import re
from complaint_categories import extract_complaints as extract_complaints_keyword
from complaint_categories_zeroshot import extract_complaints_zeroshot

# Load model and preprocessing objects
try:
    model = joblib.load('best_ensemble_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    try:
        svd = joblib.load('svd.joblib')
    except FileNotFoundError:
        svd = None
    label_encoder = joblib.load('label_encoder.joblib')
except Exception as e:
    print(f"Error loading model files: {str(e)}")
    raise

# Text cleaning for the rating prediction model
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

def count_complaints_by_category(texts, method='zeroshot', threshold=0.5):
    """
    Count complaints by category across multiple texts.
    Returns a dictionary with category names as keys and counts as values.
    
    Args:
        texts: List of review texts
        method: 'zeroshot' or 'keyword'
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
        if method == 'zeroshot':
            complaints = extract_complaints_zeroshot(text, threshold=threshold)
        else:
            complaints = extract_complaints_keyword(text)
        
        for category in complaints:
            complaint_counts[category] += 1
    
    return complaint_counts

def format_complaints_for_display(complaints, method_name):
    """Format complaints for display with method name"""
    if not complaints:
        return f"{method_name}: No complaints detected"
    
    formatted = f"{method_name}:\n"
    if method_name == "Keyword Method":
        # For keyword method, complaints are simpler
        for category in complaints:
            formatted_category = category.replace('_', ' ').title()
            formatted += f"- {formatted_category}\n"
    else:
        # For zero-shot method, include confidence scores
        sorted_complaints = sorted(complaints.items(), key=lambda item: item[1]['score'], reverse=True)
        for category, info in sorted_complaints:
            formatted_category = category.replace('_', ' ').title()
            formatted += f"- {formatted_category}: Confidence {info['score']:.2f}\n"
    
    return formatted

def predict_rating_and_complaints_comparison(text):
    """
    Predict rating and analyze complaints using both methods for comparison.
    Returns the predicted rating and both complaint analyses.
    """
    try:
        # Use cleaned text for the trained rating model
        clean_input = clean_text(text)
        
        # Use original, un-cleaned text for the zero-shot model for better context
        original_text = text
        
        X = vectorizer.transform([clean_input])
        if svd is not None:
            X = svd.transform(X)
        else:
            X = X.toarray()
        
        # Add lexicon and meta features
        lexicon_features = extract_lexicon_features([clean_input])
        meta_features = extract_meta_features([clean_input])
        X_full = np.hstack([X, lexicon_features, meta_features])
        
        # Make rating prediction
        pred = model.predict(X_full)
        predicted_rating = label_encoder.inverse_transform(pred)[0]
        
        # Get complaints using both methods
        keyword_complaints = extract_complaints_keyword(original_text)
        zeroshot_complaints = extract_complaints_zeroshot(original_text, threshold=0.5)
        
        # Format complaints for display
        keyword_text = format_complaints_for_display(keyword_complaints, "Keyword Method")
        zeroshot_text = format_complaints_for_display(zeroshot_complaints, "Zero-Shot Method")
        
        return predicted_rating, keyword_text, zeroshot_text
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return 0, f"Error occurred: {str(e)}", f"Error occurred: {str(e)}"

# Export the functions
__all__ = ['predict_rating_and_complaints_comparison', 'count_complaints_by_category']

# Only create the interface if this file is run directly
if __name__ == "__main__":
    iface = gr.Interface(
        fn=predict_rating_and_complaints_comparison,
        inputs=gr.Textbox(
            label="Enter your review text",
            placeholder="Type your review here...",
            lines=5
        ),
        outputs=[
            gr.Number(label="Predicted Rating (1-5)"),
            gr.Textbox(label="Keyword-Based Complaint Analysis", lines=6),
            gr.Textbox(label="Zero-Shot Complaint Analysis", lines=6)
        ],
        title="Review Analysis Dashboard - Method Comparison",
        description="Compare keyword-based vs zero-shot complaint detection methods. Enter a product review to see how both methods perform.",
        examples=[
            ["The product arrived quickly and fits perfectly. Very satisfied with the quality!"],
            ["Poor quality material, arrived damaged and doesn't fit well. Would not recommend."],
            ["Good product but shipping took longer than expected. The material quality is decent."],
            ["The sound is very good but it isn't cheap and the shipping was late."],
            ["bad material quality, fast shipping, affordable, seller's service is the worst"]
        ]
    )
    iface.launch(share=True) 