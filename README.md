# Product Review Rating Prediction Model

This package contains a trained ensemble model and all necessary preprocessing tools to predict product review ratings from text comments.

## Files
- `best_ensemble_model.joblib`: Trained ensemble model
- `vectorizer.joblib`: TF-IDF vectorizer
- `svd.joblib`: SVD transformer (for dimensionality reduction, if used)
- `label_encoder.joblib`: Label encoder for mapping class labels
- `inference.py`: Script for loading the model and predicting ratings from comments
- `complaint_categories.py`: Module for analyzing and categorizing complaints in reviews
- `colabdashboard.py`: Interactive Gradio dashboard for review analysis
- `requirements.txt`: List of required Python packages

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Ensure all `.joblib` files and Python scripts are in the same directory.**

## Usage
You can use the `inference.py` script to predict ratings and analyze complaints for new comments:

```bash
python inference.py
```

Or import the functions in your backend code:

```python
from inference import predict_rating_and_complaints

comments = [
    "This product is amazing!",
    "Terrible quality, very disappointed."
]
predictions, top_complaints = predict_rating_and_complaints(comments)
print("Predictions:", predictions)
print("\nTop Complaints:", top_complaints)
```

For an interactive web interface, run the Gradio dashboard:

```bash
python colabdashboard.py
```

## How it works
- The script cleans and preprocesses the input comments using the same logic as the training pipeline.
- It extracts TF-IDF, SVD, lexicon, and meta features.
- The ensemble model predicts the rating, and the label encoder maps it back to the original class label.
- The complaint analysis module categorizes and identifies key issues mentioned in the reviews.

## Complaint Analysis
The system can identify and categorize common complaints in product reviews across several categories:
- Material Quality
- Sound Quality
- Battery Life
- Comfort & Fit
- Connectivity
- Shipping & Delivery
- Price & Value
- Customer Service

Each category has specific keywords and patterns that help identify relevant complaints. The analysis returns the top complaints found in the reviews along with their descriptions.

## Notes
- If you retrain the model, regenerate and replace the `.joblib` files.
- For best results, use the same preprocessing and feature extraction as in `inference.py`.
- The complaint categories can be customized by modifying the keywords in `complaint_categories.py`.
