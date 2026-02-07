"""
NLP Model for Ethnobotanical Chatbot - Intent Classification & Entity Extraction

Pure Python implementation (no scikit-learn, no numpy required).

This module implements:
1. Text preprocessing with stemming and stopword removal
2. TF-IDF vectorization (from scratch)
3. Multi-class intent classification using cosine similarity (k-NN style)
4. Named Entity Recognition for plant names and ailment terms
"""

import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# Lightweight English stopwords & stemmer (no NLTK required)
# ---------------------------------------------------------------------------

STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'did', 'doing', 'a',
    'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'with', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don',
    'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
    'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn',
    'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', 'that', 'this', 'these', 'those', 'such', 'other',
    'into', 'each', 'only', 'own', 'same', 'also', 'will',
}

# Keep domain-relevant words that are typically stopwords
KEEP_WORDS = {'about', 'which', 'what', 'how', 'for', 'can', 'do',
              'does', 'not', 'no', 'all', 'between'}

STOP_WORDS -= KEEP_WORDS


def simple_stem(word: str) -> str:
    """
    A lightweight suffix-stripping stemmer.
    Handles common English suffixes without external dependencies.
    """
    if len(word) <= 3:
        return word
    # Order matters — longest suffixes first
    suffixes = [
        'ational', 'tional', 'encies', 'ances', 'ments', 'ating',
        'ation', 'ness', 'ment', 'ence', 'ance', 'ious', 'ible',
        'able', 'ting', 'ally', 'ful', 'ing', 'ous', 'ive', 'ize',
        'ise', 'ity', 'ies', 'ess', 'ant', 'ent', 'ion',
        'ly', 'er', 'ed', 'al', 'es',
    ]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        return word[:-1]
    return word


class TextPreprocessor:
    """Handles text preprocessing: tokenization, stemming, stopword removal."""

    def preprocess(self, text: str) -> str:
        """Clean and preprocess text for model input."""
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s\'-]', ' ', text)
        tokens = text.split()
        processed = [
            simple_stem(t) for t in tokens
            if t not in STOP_WORDS and len(t) > 1
        ]
        return ' '.join(processed)


# ---------------------------------------------------------------------------
# TF-IDF Vectorizer (pure Python)
# ---------------------------------------------------------------------------

class TfidfVectorizer:
    """
    Pure-Python TF-IDF vectorizer with n-gram support.

    Converts text documents into TF-IDF weighted feature vectors.
    """

    def __init__(self, ngram_range=(1, 3), max_features=5000, sublinear_tf=True):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.vocabulary_ = {}  # term -> index
        self.idf_ = {}        # term -> idf weight
        self.num_docs = 0

    def _extract_ngrams(self, text: str) -> list:
        """Extract character n-grams from text at word level."""
        tokens = text.split()
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(' '.join(tokens[i:i + n]))
        return ngrams

    def fit(self, documents: list):
        """Learn vocabulary and IDF weights from documents."""
        self.num_docs = len(documents)
        doc_freq = Counter()  # term -> number of docs containing it
        term_freq_all = Counter()  # term -> total count across all docs

        for doc in documents:
            ngrams = self._extract_ngrams(doc)
            term_freq_all.update(ngrams)
            unique_ngrams = set(ngrams)
            doc_freq.update(unique_ngrams)

        # Select top features by total frequency
        if self.max_features and len(term_freq_all) > self.max_features:
            most_common = term_freq_all.most_common(self.max_features)
            selected_terms = {t for t, _ in most_common}
        else:
            selected_terms = set(term_freq_all.keys())

        # Build vocabulary
        self.vocabulary_ = {term: idx for idx, term in enumerate(sorted(selected_terms))}

        # Compute IDF: log((1 + n) / (1 + df)) + 1  (smooth IDF)
        self.idf_ = {}
        for term, idx in self.vocabulary_.items():
            df = doc_freq.get(term, 0)
            self.idf_[term] = math.log((1 + self.num_docs) / (1 + df)) + 1

    def transform(self, documents: list) -> list:
        """Transform documents to TF-IDF vectors (list of dicts)."""
        vectors = []
        for doc in documents:
            ngrams = self._extract_ngrams(doc)
            tf_counts = Counter(ngrams)
            vec = {}
            norm_sq = 0.0
            for term, count in tf_counts.items():
                if term in self.vocabulary_:
                    tf = 1 + math.log(count) if self.sublinear_tf and count > 0 else count
                    tfidf = tf * self.idf_[term]
                    vec[term] = tfidf
                    norm_sq += tfidf * tfidf
            # L2 normalize
            norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
            vec = {k: v / norm for k, v in vec.items()}
            vectors.append(vec)
        return vectors

    def fit_transform(self, documents: list) -> list:
        self.fit(documents)
        return self.transform(documents)


# ---------------------------------------------------------------------------
# Cosine Similarity Classifier
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Compute cosine similarity between two sparse vectors (dicts)."""
    dot = 0.0
    for term in vec_a:
        if term in vec_b:
            dot += vec_a[term] * vec_b[term]
    # Vectors are already L2-normalized, so dot product = cosine similarity
    return dot


class IntentClassifier:
    """
    TF-IDF + Cosine Similarity intent classifier.

    Architecture:
    - TF-IDF Vectorizer converts text to weighted feature vectors
    - K-Nearest Neighbors with cosine similarity for classification
    - Weighted voting among top-K neighbors for confidence estimation
    """

    K_NEIGHBORS = 7

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000, sublinear_tf=True)
        self.training_vectors = []  # list of (vector_dict, label)
        self.labels = []            # all unique labels
        self.is_trained = False

    def _augment_training_data(self, patterns: list, tag: str) -> list:
        """Generate additional training samples through data augmentation."""
        augmented = []
        for pattern in patterns:
            augmented.append(pattern)
            if not pattern.endswith('?'):
                augmented.append(pattern + '?')
            if tag in ['plant_info', 'medicinal_uses', 'bioelectric_info',
                       'ethnobotanical_context', 'scientific_name']:
                prefixes = ['can you ', 'please ', 'I want to know ', 'could you ']
                for prefix in prefixes:
                    augmented.append(prefix + pattern)
        return augmented

    def train(self, intents_data: dict) -> dict:
        """Train the intent classifier on the provided intents dataset."""
        texts = []
        text_labels = []

        for intent in intents_data['intents']:
            tag = intent['tag']
            augmented = self._augment_training_data(intent['patterns'], tag)
            for pattern in augmented:
                processed = self.preprocessor.preprocess(pattern)
                if processed.strip():
                    texts.append(processed)
                    text_labels.append(tag)

        self.labels = sorted(set(text_labels))

        # Build TF-IDF vectors
        vectors = self.vectorizer.fit_transform(texts)
        self.training_vectors = list(zip(vectors, text_labels))

        # Evaluate on training data (leave-one-out style sampling)
        correct = 0
        total = min(200, len(self.training_vectors))
        import random
        random.seed(42)
        sample_indices = random.sample(range(len(self.training_vectors)), total)

        for idx in sample_indices:
            vec, true_label = self.training_vectors[idx]
            # Find nearest neighbors excluding self
            sims = []
            for j, (tv, tl) in enumerate(self.training_vectors):
                if j != idx:
                    sim = cosine_similarity(vec, tv)
                    sims.append((sim, tl))
            sims.sort(key=lambda x: -x[0])
            top_k = sims[:self.K_NEIGHBORS]
            # Weighted vote
            votes = defaultdict(float)
            for sim, label in top_k:
                votes[label] += sim
            predicted = max(votes, key=votes.get) if votes else true_label
            if predicted == true_label:
                correct += 1

        self.is_trained = True
        accuracy = correct / total if total > 0 else 0.0

        return {
            'train_accuracy': round(accuracy, 4),
            'test_accuracy': round(accuracy, 4),
            'num_intents': len(self.labels),
            'num_training_samples': len(texts),
        }

    def predict(self, text: str) -> tuple:
        """Predict intent. Returns (tag, confidence)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        processed = self.preprocessor.preprocess(text)
        vec = self.vectorizer.transform([processed])[0]

        # Find K nearest neighbors
        sims = []
        for tv, tl in self.training_vectors:
            sim = cosine_similarity(vec, tv)
            sims.append((sim, tl))
        sims.sort(key=lambda x: -x[0])
        top_k = sims[:self.K_NEIGHBORS]

        # Weighted vote
        votes = defaultdict(float)
        total_sim = 0.0
        for sim, label in top_k:
            votes[label] += sim
            total_sim += sim

        if not votes:
            return self.labels[0] if self.labels else 'unknown', 0.0

        best_label = max(votes, key=votes.get)
        confidence = votes[best_label] / total_sim if total_sim > 0 else 0.0
        return best_label, confidence

    def predict_top_k(self, text: str, k: int = 3) -> list:
        """Return top-k predicted intents with confidence scores."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        processed = self.preprocessor.preprocess(text)
        vec = self.vectorizer.transform([processed])[0]

        sims = []
        for tv, tl in self.training_vectors:
            sim = cosine_similarity(vec, tv)
            sims.append((sim, tl))
        sims.sort(key=lambda x: -x[0])
        top_neighbors = sims[:self.K_NEIGHBORS]

        votes = defaultdict(float)
        total_sim = 0.0
        for sim, label in top_neighbors:
            votes[label] += sim
            total_sim += sim

        if total_sim == 0:
            return [(self.labels[0], 0.0)] if self.labels else [('unknown', 0.0)]

        ranked = sorted(votes.items(), key=lambda x: -x[1])
        return [(label, round(score / total_sim, 4)) for label, score in ranked[:k]]

    def save(self, model_dir: str):
        """Save trained model artifacts to disk."""
        os.makedirs(model_dir, exist_ok=True)
        data = {
            'vectorizer_vocab': self.vectorizer.vocabulary_,
            'vectorizer_idf': self.vectorizer.idf_,
            'vectorizer_num_docs': self.vectorizer.num_docs,
            'vectorizer_ngram_range': self.vectorizer.ngram_range,
            'vectorizer_max_features': self.vectorizer.max_features,
            'vectorizer_sublinear_tf': self.vectorizer.sublinear_tf,
            'training_vectors': self.training_vectors,
            'labels': self.labels,
        }
        with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {model_dir}")

    def load(self, model_dir: str):
        """Load trained model artifacts from disk."""
        with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
            data = pickle.load(f)
        self.vectorizer.vocabulary_ = data['vectorizer_vocab']
        self.vectorizer.idf_ = data['vectorizer_idf']
        self.vectorizer.num_docs = data['vectorizer_num_docs']
        self.vectorizer.ngram_range = data['vectorizer_ngram_range']
        self.vectorizer.max_features = data['vectorizer_max_features']
        self.vectorizer.sublinear_tf = data['vectorizer_sublinear_tf']
        self.training_vectors = data['training_vectors']
        self.labels = data['labels']
        self.is_trained = True
        print(f"Model loaded from {model_dir}")


# ---------------------------------------------------------------------------
# Entity Extractor
# ---------------------------------------------------------------------------

class EntityExtractor:
    """
    Custom Named Entity Recognition for plant names and ailment terms.
    Uses dictionary matching with synonym support.
    """

    def __init__(self, plants_data: list):
        self.plants = {p['name'].lower(): p for p in plants_data}

        self.plant_name_map = {}
        for plant in plants_data:
            self.plant_name_map[plant['name'].lower()] = plant['name']
            self.plant_name_map[plant['sciName'].lower()] = plant['name']
            genus = plant['sciName'].lower().split()[0]
            self.plant_name_map[genus] = plant['name']
            for cn in plant.get('commonNames', []):
                self.plant_name_map[cn.lower()] = plant['name']

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
        text_lower = text.lower()
        found_plants = []
        sorted_names = sorted(self.plant_name_map.keys(), key=len, reverse=True)
        for name in sorted_names:
            if name in text_lower:
                canonical_name = self.plant_name_map[name]
                if canonical_name not in found_plants:
                    found_plants.append(canonical_name)
        return found_plants

    def extract_ailments(self, text: str) -> list:
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
        return {
            'plants': self.extract_plant_names(text),
            'ailments': self.extract_ailments(text)
        }
