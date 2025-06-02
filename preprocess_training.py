"""
Untitled-2.py (now: fast, modular text classification pipeline)

A fast, modular text classification pipeline for sentiment/review analysis.
- Cleans and preprocesses text data.
- Uses TF-IDF (up to 3-grams) and optional SVD.
- Models: XGBoost, RandomForest, LightGBM, CatBoost, LinearSVC (no Logistic Regression).
- Voting ensemble for accuracy and speed.
- Hyperparameter tuning for XGBoost/RandomForest.
- Prints accuracy, classification report, confusion matrix, and top features.
- Saves best model to disk.

To add/remove models or change parameters, edit the MODEL_CONFIG section below.
"""

import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
from imblearn.over_sampling import SMOTE
import sys

# Try to import XGBoost, LightGBM, CatBoost and check for GPU support
try:
    from xgboost import XGBClassifier
    import xgboost
    XGB_AVAILABLE = True
    # Check for GPU support in XGBoost
    try:
        gpu_params = {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}
        # Try a dummy fit to check GPU
        dummy_xgb = XGBClassifier(**gpu_params, use_label_encoder=False, eval_metric='mlogloss')
        dummy_xgb.fit(np.array([[0],[1]]), np.array([0,1]))
        XGB_GPU = True
        print("XGBoost: GPU is available and will be used.")
    except Exception as e:
        XGB_GPU = False
        print(f"XGBoost: GPU not available, falling back to CPU. Reason: {e}")
except ImportError:
    XGB_AVAILABLE = False
    XGB_GPU = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
    # Check for GPU support in LightGBM
    try:
        dummy_lgbm = LGBMClassifier(device='gpu')
        dummy_lgbm.fit(np.array([[0],[1]]), np.array([0,1]))
        LGBM_GPU = True
        print("LightGBM: GPU is available and will be used.")
    except Exception as e:
        LGBM_GPU = False
        print(f"LightGBM: GPU not available, falling back to CPU. Reason: {e}")
except ImportError:
    LGBM_AVAILABLE = False
    LGBM_GPU = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
    # Check for GPU support in CatBoost
    try:
        dummy_cat = CatBoostClassifier(task_type='GPU', devices='0', verbose=0)
        dummy_cat.fit(np.array([[0],[1]]), np.array([0,1]))
        CATBOOST_GPU = True
        print("CatBoost: GPU is available and will be used.")
    except Exception as e:
        CATBOOST_GPU = False
        print(f"CatBoost: GPU not available, falling back to CPU. Reason: {e}")
except ImportError:
    CATBOOST_AVAILABLE = False
    CATBOOST_GPU = False

# If running in Google Colab, upgrade lightgbm to ensure early stopping works
if 'google.colab' in sys.modules:
    try:
        import lightgbm
        from packaging import version
        if version.parse(lightgbm.__version__) < version.parse('2.2.1'):
            print('Upgrading lightgbm...')
            import os
            os.system('pip install --upgrade lightgbm')
    except ImportError:
        print('Installing lightgbm...')
        import os
        os.system('pip install lightgbm')

# ========== USER CONFIGURABLE SECTION ==========
N = 50000  # Number of samples to use
RANDOM_STATE = 42
USE_SVD = True  # Use only SVD (no raw TF-IDF)
SVD_COMPONENTS = 300  # Increased number of SVD components for more feature variation

MODEL_CONFIG = [
    ("xgboost", XGB_AVAILABLE),
    ("lightgbm", LGBM_AVAILABLE),
]

# ========== DATA LOADING & CLEANING ==========
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'[\W_]+', ' ', text)  # Remove non-word chars
    text = text.lower()
    return text

def load_and_prepare_data(path='Reviews.csv', n_samples=N, random_state=RANDOM_STATE, balance_classes=True):
    df = pd.read_csv(path)
    # Filter out reviews with HelpfulnessDenominator == 0
    if 'HelpfulnessDenominator' in df.columns:
        df = df[df['HelpfulnessDenominator'] > 0]
    # Remove duplicate reviews (same UserId, ProductId, and Text)
    if set(['UserId', 'ProductId', 'Text']).issubset(df.columns):
        df = df.drop_duplicates(subset=['UserId', 'ProductId', 'Text'])
    # Concatenate Summary and Text into 'full_text' if both exist
    if set(['Summary', 'Text']).issubset(df.columns):
        df['full_text'] = df['Summary'].astype(str) + ' ' + df['Text'].astype(str)
        text_col = 'full_text'
    else:
        # Try to find the text column as before
        possible_text_cols = ['review', 'Review', 'text', 'Text', 'comment', 'Comment',
                              'description', 'Description', 'reviewText', 'summary']
        text_col = next((col for col in possible_text_cols if col in df.columns), df.columns[0])
    label_col = 'Score' if 'Score' in df.columns else df.columns[-1]
    df = df.dropna(subset=[text_col, label_col])
    if balance_classes:
        # Get class counts
        class_counts = df[label_col].value_counts()
        per_class_n = min(20000, class_counts.min())  # Take up to 20000 per class, or as many as available
        balanced_df = []
        for cls in class_counts.index:
            cls_df = df[df[label_col] == cls].sample(n=min(per_class_n, len(df[df[label_col] == cls])), random_state=random_state)
            balanced_df.append(cls_df)
        df = pd.concat(balanced_df).sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
    df['clean_text'] = df[text_col].apply(clean_text)
    return df, text_col, label_col

def extract_features(df, use_svd=USE_SVD, svd_components=SVD_COMPONENTS):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,3), sublinear_tf=True)
    X = vectorizer.fit_transform(df['clean_text'])
    if use_svd:
        svd = TruncatedSVD(n_components=svd_components, random_state=RANDOM_STATE)
        X = svd.fit_transform(X)
        feature_names = [f'svd_{i}' for i in range(svd_components)]
    else:
        svd = None
        feature_names = vectorizer.get_feature_names_out()
    return X, feature_names, vectorizer, svd

def extract_lexicon_features(texts):
    """
    Extracts lexicon-based sentiment features for each text:
    - positive word count
    - negative word count
    - intensifier count
    - sentiment score (pos - neg)
    """
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
    # Generate negated positive terms
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
    """
    Extracts additional meta features for each text:
    - review length
    - avg word length
    - punctuation count
    - uppercase word count
    """
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

# ========== MODEL DEFINITIONS ==========
def get_models():
    models = []
    if ("xgboost", True) in MODEL_CONFIG and XGB_AVAILABLE:
        if XGB_GPU:
            print("Using XGBoost with GPU.")
            models.append(('xgboost', XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8,
                colsample_bytree=0.8, use_label_encoder=False, eval_metric='mlogloss',
                n_jobs=-1, random_state=RANDOM_STATE, tree_method='gpu_hist', predictor='gpu_predictor')))
        else:
            print("Using XGBoost with CPU.")
            models.append(('xgboost', XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8,
                colsample_bytree=0.8, use_label_encoder=False, eval_metric='mlogloss',
                n_jobs=-1, random_state=RANDOM_STATE, tree_method='auto', predictor='auto')))
    if ("lightgbm", True) in MODEL_CONFIG and LGBM_AVAILABLE:
        if LGBM_GPU:
            print("Using LightGBM with GPU.")
            models.append(('lightgbm', LGBMClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8,
                colsample_bytree=0.8, n_jobs=-1, random_state=RANDOM_STATE, device='gpu', verbosity=-1)))
        else:
            print("Using LightGBM with CPU.")
            models.append(('lightgbm', LGBMClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8,
                colsample_bytree=0.8, n_jobs=-1, random_state=RANDOM_STATE, device='cpu', verbosity=-1)))
    return models

# ========== MAIN PIPELINE ==========
def main():
    print("Loading and cleaning data...")
    df, text_col, label_col = load_and_prepare_data(balance_classes=True)
    print(f"Using text column: {text_col}, label column: {label_col}")
    print("Extracting features...")
    X, feature_names, vectorizer, svd = extract_features(df, use_svd=USE_SVD, svd_components=SVD_COMPONENTS)
    # Add lexicon-based features
    lexicon_features = extract_lexicon_features(df['clean_text'])
    # Add meta features
    meta_features = extract_meta_features(df['clean_text'])
    X = np.hstack([X if isinstance(X, np.ndarray) else X.toarray(), lexicon_features, meta_features])
    feature_names = list(feature_names) + ['pos_count', 'neg_count', 'intensifier_count', 'sentiment_score',
                                           'review_length', 'avg_word_length', 'punct_count', 'upper_count']
    y = df[label_col].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc)

    # --- No SMOTE: Use only class_weight='balanced' ---

    print("Defining models...")
    models = get_models()
    print(f"Models in ensemble: {[name for name, _ in models]}")

    # Hyperparameter tuning for XGBoost and LightGBM (faster search)
    tuned_models = []
    for name, model in models:
        if name == 'xgboost':
            param_dist = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0],
            }
            search = RandomizedSearchCV(model, param_dist, n_iter=5, scoring='accuracy',
                                       n_jobs=-1, cv=2, random_state=RANDOM_STATE, verbose=1)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            print(f"Best XGBoost params: {search.best_params_}")
            # Refit with early stopping on the full training set (remove early_stopping_rounds for XGBoost)
            best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            tuned_models.append(('xgboost', best_model))
        elif name == 'lightgbm':
            param_dist = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0],
                'min_data_in_leaf': [5, 10, 20],  # Lowered for more splits
                'min_data_in_bin': [1, 5, 10],     # Lowered for more splits
            }
            search = RandomizedSearchCV(model, param_dist, n_iter=5, scoring='accuracy',
                                       n_jobs=-1, cv=2, random_state=RANDOM_STATE, verbose=1)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            print(f"Best LightGBM params: {search.best_params_}")
            try:
                best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, eval_metric='logloss')
            except TypeError as e:
                print(f"Warning: {e}. Retrying without early_stopping_rounds and eval_metric.")
                best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
            tuned_models.append(('lightgbm', best_model))
        else:
            tuned_models.append((name, model))

    # Voting ensemble
    all_proba = all(hasattr(m, 'predict_proba') for _, m in tuned_models)
    voting = 'soft' if all_proba else 'hard'
    ensemble = VotingClassifier(estimators=tuned_models, voting=voting, n_jobs=-1)
    print(f"Training VotingClassifier with voting='{voting}'...")
    ensemble.fit(X_train, y_train)

    print("\nEvaluating ensemble...")
    y_pred = ensemble.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_.astype(str)))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the best model
    joblib.dump(ensemble, 'best_ensemble_model.joblib')
    print("\nBest ensemble model saved as 'best_ensemble_model.joblib'.")
    # Save vectorizer, svd, and label encoder for dashboard use
    joblib.dump(vectorizer, 'vectorizer.joblib')
    if svd is not None:
        joblib.dump(svd, 'svd.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    print("Vectorizer, SVD, and label encoder saved for dashboard use.")

    # Print top 20 most important features from the best model if available
    for name, model in tuned_models:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[-20:][::-1]
            print(f"\nTop 20 features for {name}:")
            for idx in top_idx:
                print(f"{feature_names[idx]}: {importances[idx]:.4f}")
            break  # Only print for the first model with importances

    # --- Optional: For binary/ternary classification, map y to fewer classes here ---
    # Example: df['label'] = df['Score'].map(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))
    # Then re-run the pipeline for binary/ternary classification

if __name__ == "__main__":
    main()