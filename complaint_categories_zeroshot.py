"""
Complaint extraction using zero-shot classification (distilbart-mnli-12-1).
- Uses GPU if available, falls back to CPU.
- Slower than keyword-based, but much more robust.
- To revert, use complaint_categories.py instead.
"""
from transformers import pipeline
import torch

COMPLAINT_LABELS = {
    "material_quality": "Bad material quality, cheap, flimsy, broke, damaged",
    "sound_quality": "Poor sound, muffled, distortion, static, bad audio",
    "battery_life": "Short battery life, battery dies quickly, charging issues",
    "comfort_fit": "Uncomfortable, too tight, too loose, painful to wear",
    "connectivity": "Connection issues, disconnects, lag, pairing problems",
    "shipping_delivery": "Late delivery, damaged packaging, lost item",
    "price_value": "Too expensive, overpriced, not worth the money",
    "customer_service": "Bad customer service, unhelpful, rude, no response"
}

# Try to use GPU if available, else CPU
if torch.cuda.is_available():
    device = 0
else:
    device = -1

zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

def extract_complaints_zeroshot(text, threshold=0.5):
    """
    Use zero-shot classification to extract complaint categories from text.
    Returns a dictionary with complaint categories and their scores.
    """
    result = zero_shot_classifier(
        text,
        list(COMPLAINT_LABELS.values()),
        multi_label=True
    )
    complaints = {}
    for label, score in zip(result['labels'], result['scores']):
        if score >= threshold:
            # Find the category key by description
            for k, v in COMPLAINT_LABELS.items():
                if v == label:
                    complaints[k] = {'score': float(score), 'description': v}
    return complaints 