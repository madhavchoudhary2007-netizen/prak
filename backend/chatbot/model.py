"""
NLP Model for Ethnobotanical Chatbot - Intent Classification & Entity Extraction

This module implements:
1. Text preprocessing with lemmatization and stopword removal
2. TF-IDF vectorization for feature extraction
3. Multi-class intent classification using SVM (Support Vector Machine)
4. Named Entity Recognition for plant names and ailment terms
"""

import json
import os
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Use NLTK for text preprocessing
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


class TextPreprocessor:
    """Handles text preprocessing: tokenization, lemmatization, stopword removal."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Keep domain-relevant words that might be in stopwords
        self.keep_words = {
            'about', 'which', 'what', 'how', 'for', 'can', 'do',
            'does', 'not', 'no', 'all', 'between'
        }
        self.stop_words -= self.keep_words

    def preprocess(self, text: str) -> str:
        """Clean and preprocess text for model input."""
        text = text.lower().strip()
        # Remove special characters but keep basic punctuation context
        text = re.sub(r'[^a-z0-9\s\'-]', ' ', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Lemmatize and remove stopwords
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]
        return ' '.join(processed_tokens)


class IntentClassifier:
    """
    SVM-based intent classifier using TF-IDF features.

    Architecture:
    - TF-IDF Vectorizer converts text to numerical feature vectors
    - Support Vector Machine (SVM) with RBF kernel for multi-class classification
    - Label Encoder for mapping intent tags to numeric labels
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),  # unigrams, bigrams, and trigrams
            sublinear_tf=True,
            min_df=1
        )
        self.classifier = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True,  # Enable probability estimates
            decision_function_shape='ovr'
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def _augment_training_data(self, patterns: list, tag: str) -> list:
        """Generate additional training samples through data augmentation."""
        augmented = []
        for pattern in patterns:
            augmented.append(pattern)
            # Add with question marks
            if not pattern.endswith('?'):
                augmented.append(pattern + '?')
            # Add with common prefixes
            if tag in ['plant_info', 'medicinal_uses', 'bioelectric_info',
                       'ethnobotanical_context', 'scientific_name']:
                prefixes = ['can you ', 'please ', 'I want to know ', 'could you ']
                for prefix in prefixes:
                    augmented.append(prefix + pattern)
        return augmented

    def train(self, intents_data: dict) -> dict:
        """
        Train the intent classifier on the provided intents dataset.

        Returns a dict with training metrics.
        """
        texts = []
        labels = []

        for intent in intents_data['intents']:
            tag = intent['tag']
            augmented_patterns = self._augment_training_data(intent['patterns'], tag)
            for pattern in augmented_patterns:
                processed = self.preprocessor.preprocess(pattern)
                if processed.strip():
                    texts.append(processed)
                    labels.append(tag)

        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)

        # Create TF-IDF features
        X = self.vectorizer.fit_transform(texts)

        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )

        # Train SVM classifier
        self.classifier.fit(X_train, y_train)

        # Evaluate
        y_pred = self.classifier.predict(X_test)
        train_accuracy = self.classifier.score(X_train, y_train)
        test_accuracy = self.classifier.score(X_test, y_test)

        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )

        self.is_trained = True

        return {
            'train_accuracy': round(train_accuracy, 4),
            'test_accuracy': round(test_accuracy, 4),
            'num_intents': len(self.label_encoder.classes_),
            'num_training_samples': len(texts),
            'classification_report': report
        }

    def predict(self, text: str) -> tuple:
        """
        Predict the intent of a given text.

        Returns: (predicted_tag, confidence_score)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        processed = self.preprocessor.preprocess(text)
        X = self.vectorizer.transform([processed])

        # Get prediction and probability
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = float(np.max(probabilities))

        tag = self.label_encoder.inverse_transform([prediction])[0]

        return tag, confidence

    def predict_top_k(self, text: str, k: int = 3) -> list:
        """
        Return top-k predicted intents with confidence scores.

        Returns: list of (tag, confidence) tuples
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        processed = self.preprocessor.preprocess(text)
        X = self.vectorizer.transform([processed])
        probabilities = self.classifier.predict_proba(X)[0]

        # Get top-k indices
        top_k_indices = np.argsort(probabilities)[::-1][:k]

        results = []
        for idx in top_k_indices:
            tag = self.label_encoder.inverse_transform([idx])[0]
            conf = float(probabilities[idx])
            results.append((tag, conf))

        return results

    def save(self, model_dir: str):
        """Save trained model artifacts to disk."""
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(model_dir, 'classifier.pkl'), 'wb') as f:
            pickle.dump(self.classifier, f)
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Model saved to {model_dir}")

    def load(self, model_dir: str):
        """Load trained model artifacts from disk."""
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(os.path.join(model_dir, 'classifier.pkl'), 'rb') as f:
            self.classifier = pickle.load(f)
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {model_dir}")


class EntityExtractor:
    """
    Custom Named Entity Recognition for plant names and ailment terms.

    Uses fuzzy matching and synonym dictionaries to extract entities
    from user queries.
    """

    def __init__(self, plants_data: list):
        self.plants = {p['name'].lower(): p for p in plants_data}

        # Build comprehensive plant name lookup (name, scientific name, common names)
        self.plant_name_map = {}
        for plant in plants_data:
            self.plant_name_map[plant['name'].lower()] = plant['name']
            self.plant_name_map[plant['sciName'].lower()] = plant['name']
            # First word of scientific name (genus)
            genus = plant['sciName'].lower().split()[0]
            self.plant_name_map[genus] = plant['name']
            # Common names
            for cn in plant.get('commonNames', []):
                self.plant_name_map[cn.lower()] = plant['name']

        # Ailment synonym dictionary
        self.ailment_map = {
            'stress': ['stress', 'stressed', 'tension', 'tense', 'worry', 'worried',
                       'anxious', 'anxiety', 'panic', 'overwhelmed', 'burnout'],
            'memory': ['memory', 'brain', 'cognitive', 'concentration', 'focus',
                       'mental', 'intelligence', 'learning', 'forgetful', 'brain fog',
                       'thinking', 'clarity', 'mind'],
            'respiratory': ['cough', 'cold', 'respiratory', 'breathing', 'lungs',
                           'asthma', 'bronchitis', 'flu', 'fever', 'sore throat',
                           'congestion', 'phlegm', 'sinusitis', 'nasal'],
            'digestion': ['digestion', 'stomach', 'digestive', 'gastric', 'acidity',
                         'nausea', 'bloating', 'constipation', 'indigestion',
                         'gut', 'appetite', 'metabolism', 'acid reflux', 'gastritis'],
            'immunity': ['immunity', 'immune', 'infection', 'defense', 'resistance',
                        'protect', 'prevention', 'disease', 'health'],
            'skin': ['skin', 'acne', 'rash', 'eczema', 'dermatitis', 'complexion',
                    'beauty', 'wound', 'scar', 'pimple', 'blemish', 'glow'],
            'inflammation': ['inflammation', 'swelling', 'joint', 'arthritis',
                           'pain', 'chronic pain', 'muscle pain', 'ache', 'sore'],
            'sleep': ['sleep', 'insomnia', 'sleepless', 'rest', 'restless',
                     'trouble sleeping', 'sedative', 'relaxation']
        }

    def extract_plant_names(self, text: str) -> list:
        """Extract plant names mentioned in the text."""
        text_lower = text.lower()
        found_plants = []

        # Sort by name length (longest first) to match multi-word names first
        sorted_names = sorted(self.plant_name_map.keys(), key=len, reverse=True)

        for name in sorted_names:
            if name in text_lower:
                canonical_name = self.plant_name_map[name]
                if canonical_name not in found_plants:
                    found_plants.append(canonical_name)

        return found_plants

    def extract_ailments(self, text: str) -> list:
        """Extract ailment categories mentioned in the text."""
        text_lower = text.lower()
        found_ailments = []

        for ailment_category, synonyms in self.ailment_map.items():
            for synonym in synonyms:
                if synonym in text_lower:
                    if ailment_category not in found_ailments:
                        found_ailments.append(ailment_category)
                    break

        return found_ailments

    def extract_all(self, text: str) -> dict:
        """Extract all entities from text."""
        return {
            'plants': self.extract_plant_names(text),
            'ailments': self.extract_ailments(text)
        }
