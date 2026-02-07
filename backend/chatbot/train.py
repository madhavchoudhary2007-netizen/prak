"""
Training Script for the Ethnobotanical Chatbot NLP Model

This script:
1. Loads training data (intents.json) and plant data (plants.json)
2. Trains the TF-IDF + SVM intent classifier
3. Evaluates model performance
4. Saves trained model artifacts to disk
"""

import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.model import IntentClassifier, EntityExtractor


def train_model():
    """Train and save the NLP model."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    model_dir = os.path.join(base_dir, 'trained_model')

    # Load training data
    print("=" * 60)
    print("ETHNOBOTANICAL CHATBOT - MODEL TRAINING")
    print("=" * 60)

    print("\n[1/4] Loading training data...")
    with open(os.path.join(data_dir, 'intents.json'), 'r') as f:
        intents_data = json.load(f)
    print(f"  Loaded {len(intents_data['intents'])} intent categories")

    total_patterns = sum(len(i['patterns']) for i in intents_data['intents'])
    print(f"  Total training patterns: {total_patterns}")

    # Load plant data
    print("\n[2/4] Loading plant database...")
    with open(os.path.join(data_dir, 'plants.json'), 'r') as f:
        plants_data = json.load(f)
    print(f"  Loaded {len(plants_data)} plants")

    # Train intent classifier
    print("\n[3/4] Training Intent Classifier (TF-IDF + SVM)...")
    classifier = IntentClassifier()
    metrics = classifier.train(intents_data)

    print(f"\n  Training Results:")
    print(f"  ├── Training Accuracy:  {metrics['train_accuracy'] * 100:.1f}%")
    print(f"  ├── Test Accuracy:      {metrics['test_accuracy'] * 100:.1f}%")
    print(f"  ├── Number of Intents:  {metrics['num_intents']}")
    print(f"  └── Training Samples:   {metrics['num_training_samples']} (after augmentation)")

    # Print per-intent performance
    print("\n  Per-Intent Performance (Test Set):")
    report = metrics['classification_report']
    for intent_name, scores in report.items():
        if isinstance(scores, dict) and 'precision' in scores:
            p = scores['precision']
            r = scores['recall']
            f1 = scores['f1-score']
            if scores.get('support', 0) > 0:
                print(f"    {intent_name:30s} P:{p:.2f}  R:{r:.2f}  F1:{f1:.2f}")

    # Save model
    print(f"\n[4/4] Saving trained model to {model_dir}...")
    classifier.save(model_dir)

    # Quick verification
    print("\n" + "=" * 60)
    print("MODEL VERIFICATION - Sample Predictions:")
    print("=" * 60)

    test_queries = [
        "Tell me about Ashwagandha",
        "Which plants help with stress?",
        "Scientific name of Tulsi",
        "What are nervine tonics?",
        "Hello",
        "Bioelectric properties of Brahmi",
        "Plants for digestion",
        "Traditional use of Neem",
        "Compare Brahmi and Shankhpushpi",
        "I can't sleep at night"
    ]

    for query in test_queries:
        tag, conf = classifier.predict(query)
        print(f"  '{query}'")
        print(f"    → Intent: {tag} (confidence: {conf:.2%})")

    # Test entity extraction
    print("\n" + "=" * 60)
    print("ENTITY EXTRACTION VERIFICATION:")
    print("=" * 60)

    entity_extractor = EntityExtractor(plants_data)
    entity_tests = [
        "Tell me about Ashwagandha and Brahmi",
        "Which plants help with stress and memory?",
        "Bioelectric properties of Bacopa monnieri",
        "I have a cough and cold, what should I use?"
    ]

    for query in entity_tests:
        entities = entity_extractor.extract_all(query)
        print(f"  '{query}'")
        print(f"    → Plants: {entities['plants']}")
        print(f"    → Ailments: {entities['ailments']}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    return metrics


if __name__ == '__main__':
    train_model()
