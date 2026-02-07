"""
Flask API Server for the Ethnobotanical AI Chatbot

Endpoints:
- POST /api/chat     - Send a message to the chatbot
- GET  /api/plants   - Get all plants
- GET  /api/plant/<name> - Get specific plant info
- GET  /api/health   - Health check
"""

import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from chatbot.model import IntentClassifier, EntityExtractor
from chatbot.engine import ChatbotEngine

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global references
chatbot_engine = None
plants_data = None


def initialize():
    """Load model and data on startup."""
    global chatbot_engine, plants_data

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    model_dir = os.path.join(base_dir, 'trained_model')

    # Load plant data
    with open(os.path.join(data_dir, 'plants.json'), 'r') as f:
        plants_data = json.load(f)

    # Load intents data
    with open(os.path.join(data_dir, 'intents.json'), 'r') as f:
        intents_data = json.load(f)

    # Load trained model
    classifier = IntentClassifier()
    if os.path.exists(os.path.join(model_dir, 'classifier.pkl')):
        classifier.load(model_dir)
        print("Loaded trained NLP model.")
    else:
        print("No trained model found. Training now...")
        classifier.train(intents_data)
        classifier.save(model_dir)
        print("Model trained and saved.")

    # Initialize entity extractor
    entity_extractor = EntityExtractor(plants_data)

    # Initialize chatbot engine
    chatbot_engine = ChatbotEngine(classifier, entity_extractor, plants_data, intents_data)
    print(f"Chatbot engine initialized with {len(plants_data)} plants.")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': chatbot_engine is not None,
        'plants_count': len(plants_data) if plants_data else 0
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat endpoint - processes user message through the NLP pipeline.

    Request body: { "message": "user query text" }
    Response: { "response": "...", "intent": "...", "confidence": 0.95, "entities": {...} }
    """
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing "message" field in request body'}), 400

    user_message = data['message'].strip()
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    result = chatbot_engine.get_response(user_message)
    return jsonify(result)


@app.route('/api/plants', methods=['GET'])
def get_plants():
    """Get all plants, optionally filtered by ailment."""
    ailment = request.args.get('ailment', None)
    search = request.args.get('search', '').lower()

    filtered = plants_data
    if ailment and ailment != 'all':
        filtered = [p for p in filtered if ailment in p.get('ailments', [])]
    if search:
        filtered = [p for p in filtered
                    if search in p['name'].lower()
                    or search in p['sciName'].lower()
                    or search in p['description'].lower()]

    return jsonify(filtered)


@app.route('/api/plant/<name>', methods=['GET'])
def get_plant(name):
    """Get detailed info for a specific plant by name."""
    name_lower = name.lower()
    for plant in plants_data:
        if plant['name'].lower() == name_lower or plant['sciName'].lower() == name_lower:
            return jsonify(plant)
    return jsonify({'error': f'Plant "{name}" not found'}), 404


# Initialize on module load
initialize()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
