# Product Review Rating Prediction Model

This package contains a trained ensemble model and all necessary preprocessing tools to predict product review ratings from text comments.

## Files
- `best_ensemble_model.joblib`: Trained ensemble model
- `vectorizer.joblib`: TF-IDF vectorizer
- `svd.joblib`: SVD transformer (for dimensionality reduction, if used)
- `label_encoder.joblib`: Label encoder for mapping class labels
- `inference.py`: Script for loading the model and predicting ratings from comments
- `requirements.txt`: List of required Python packages

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Ensure all `.joblib` files and `inference.py` are in the same directory.**

## Usage
You can use the `inference.py` script to predict ratings for new comments:

```bash
python inference.py
```

Or import the `predict_rating` function in your backend code:

```python
from inference import predict_rating

comments = [
    "This product is amazing!",
    "Terrible quality, very disappointed."
]
predictions = predict_rating(comments)
print(predictions)
```

## How it works
- The script cleans and preprocesses the input comments using the same logic as the training pipeline.
- It extracts TF-IDF, SVD, lexicon, and meta features.
- The ensemble model predicts the rating, and the label encoder maps it back to the original class label.

## Notes
- If you retrain the model, regenerate and replace the `.joblib` files.
- For best results, use the same preprocessing and feature extraction as in `inference.py`.
