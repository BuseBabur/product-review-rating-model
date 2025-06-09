"""
Complaint category definitions and extraction functions for product reviews.
"""

# Define complaint categories and their associated keywords
COMPLAINT_CATEGORIES = {
    'material_quality': {
        'keywords': {
            'negative': ['cheap', 'flimsy', 'break', 'broke', 'broken', 'crack', 'cracked', 'damage', 
                        'damaged', 'poor quality', 'low quality', 'cheap material', 'plastic', 'fragile',
                        'bad quality', 'terrible quality', 'awful quality']
        },
        'description': 'Issues related to the physical quality and durability of materials'
    },
    'sound_quality': {
        'keywords': {
            'negative': ['muffled', 'distortion', 'static', 'crackle', 'hiss', 'poor sound',
                        'bad audio', 'noisy', 'echo', 'terrible sound', 'awful sound']
        },
        'description': 'Issues related to audio performance and sound characteristics'
    },
    'battery_life': {
        'keywords': {
            'negative': ['drain', 'draining', 'dies quickly', 'short battery', 'battery dies', 
                        'low battery', 'battery drain', 'doesn\'t last', 'bad battery', 'terrible battery']
        },
        'description': 'Issues related to battery performance and charging'
    },
    'comfort_fit': {
        'keywords': {
            'negative': ['uncomfortable', 'uncomfy', 'tight', 'loose', 'pressure', 'pain', 'hurt', 
                        'sore', 'doesn\'t fit', 'poor fit', 'bad fit', 'terrible fit']
        },
        'description': 'Issues related to physical comfort and fit'
    },
    'connectivity': {
        'keywords': {
            'negative': ['disconnect', 'disconnected', 'drop', 'dropping', 'lag', 'latency', 
                        'delay', 'connection issues', 'poor connection', 'bad connection', 'terrible connection']
        },
        'description': 'Issues related to wireless connectivity and pairing'
    },
    'shipping_delivery': {
        'keywords': {
            'negative': ['late', 'delay', 'delayed', 'damaged', 'missing', 'lost', 'poor packaging',
                        'slow shipping', 'took long', 'bad shipping', 'terrible shipping', 'awful shipping']
        },
        'description': 'Issues related to shipping, delivery, and packaging'
    },
    'price_value': {
        'keywords': {
            'negative': ['expensive', 'overpriced', 'rip-off', 'waste', 'wasted', 'not worth',
                        'poor value', 'too expensive', 'bad value', 'terrible value']
        },
        'description': 'Issues related to pricing and value for money'
    },
    'customer_service': {
        'keywords': {
            'negative': ['unhelpful', 'unresponsive', 'poor service', 'no response', 'ignored',
                        'bad service', 'rude', 'terrible service', 'awful service']
        },
        'description': 'Issues related to customer service and support'
    }
}

def extract_complaints(text):
    """
    Extract complaint categories from a text by looking for negative keywords.
    Returns a dictionary with complaint categories and their scores.
    """
    text = text.lower()
    complaints = {}
    
    for category, info in COMPLAINT_CATEGORIES.items():
        # Check for negative keywords
        for keyword in info['keywords']['negative']:
            if keyword in text:
                complaints[category] = {
                    'score': 1,
                    'description': info['description']
                }
                break  # Found a complaint for this category, no need to check other keywords
    
    return complaints

def get_top_complaints(texts, top_n=3):
    """
    Get the most common complaints across multiple texts.
    Returns a list of tuples (category, count, description).
    """
    all_complaints = {}
    
    for text in texts:
        complaints = extract_complaints(text)
        for category, info in complaints.items():
            if category not in all_complaints:
                all_complaints[category] = {
                    'count': 0,
                    'description': info['description']
                }
            all_complaints[category]['count'] += 1
    
    # Sort by count and get top N
    sorted_complaints = sorted(
        [(cat, info['count'], info['description']) 
         for cat, info in all_complaints.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_complaints[:top_n] 