"""
Chatbot Response Engine

This module generates appropriate responses based on:
1. Predicted intent from the NLP classifier
2. Extracted entities (plants, ailments)
3. Plant database knowledge
"""

import json
import random
from typing import Optional


class ChatbotEngine:
    """
    Response generation engine that combines intent classification,
    entity extraction, and knowledge retrieval.
    """

    CONFIDENCE_THRESHOLD = 0.25  # Below this, use fallback

    def __init__(self, intent_classifier, entity_extractor, plants_data: list, intents_data: dict):
        self.classifier = intent_classifier
        self.entity_extractor = entity_extractor
        self.plants = {p['name'].lower(): p for p in plants_data}
        self.plants_list = plants_data
        self.intents_data = intents_data

        # Pre-build ailment -> plant mapping
        self.ailment_plant_map = {}
        for plant in plants_data:
            for ailment in plant.get('ailments', []):
                if ailment not in self.ailment_plant_map:
                    self.ailment_plant_map[ailment] = []
                self.ailment_plant_map[ailment].append(plant)

        # Nervine plants
        self.nervine_plants = [
            p for p in plants_data
            if p.get('nervineType') and p['nervineType'] not in [None, 'null', '']
        ]

    def get_response(self, user_input: str) -> dict:
        """
        Generate a response for the user's input.

        Returns a dict with:
        - response: the text response
        - intent: detected intent tag
        - confidence: model confidence score
        - entities: extracted entities
        """
        # Step 1: Classify intent
        intent_tag, confidence = self.classifier.predict(user_input)
        top_intents = self.classifier.predict_top_k(user_input, k=3)

        # Step 2: Extract entities
        entities = self.entity_extractor.extract_all(user_input)

        # Step 3: Use entities to refine intent if needed
        intent_tag, confidence = self._refine_intent(
            intent_tag, confidence, entities, user_input
        )

        # Step 4: Generate response based on intent + entities
        response = self._generate_response(intent_tag, confidence, entities, user_input)

        return {
            'response': response,
            'intent': intent_tag,
            'confidence': round(confidence, 4),
            'entities': entities,
            'top_intents': [{'tag': t, 'confidence': round(c, 4)} for t, c in top_intents]
        }

    def _refine_intent(self, intent_tag: str, confidence: float,
                       entities: dict, user_input: str) -> tuple:
        """
        Refine the predicted intent using entity information.
        If entities strongly suggest a different intent, override.
        """
        plants_found = entities.get('plants', [])
        ailments_found = entities.get('ailments', [])
        lower_input = user_input.lower()

        # If plant names found and intent is generic, upgrade to plant_info
        if plants_found and intent_tag in ['greeting', 'help', 'thanks', 'how_are_you']:
            if any(kw in lower_input for kw in ['bioelectric', 'nervine', 'nerve', 'electric', 'impulse']):
                return 'bioelectric_info', max(confidence, 0.7)
            if any(kw in lower_input for kw in ['use', 'medicinal', 'benefit', 'heal', 'treat', 'cure']):
                return 'medicinal_uses', max(confidence, 0.7)
            if any(kw in lower_input for kw in ['traditional', 'tribe', 'folk', 'cultural', 'community', 'ethnobotanical']):
                return 'ethnobotanical_context', max(confidence, 0.7)
            if any(kw in lower_input for kw in ['scientific', 'botanical', 'latin', 'binomial']):
                return 'scientific_name', max(confidence, 0.7)
            return 'plant_info', max(confidence, 0.7)

        # If ailments found and intent is generic, upgrade to ailment_query
        if ailments_found and intent_tag in ['greeting', 'help', 'thanks', 'how_are_you']:
            return 'ailment_query', max(confidence, 0.6)

        return intent_tag, confidence

    def _generate_response(self, intent_tag: str, confidence: float,
                           entities: dict, user_input: str) -> str:
        """Generate response based on intent and entities."""

        # Low confidence fallback
        if confidence < self.CONFIDENCE_THRESHOLD:
            return self._fallback_response(entities)

        plants_found = entities.get('plants', [])
        ailments_found = entities.get('ailments', [])

        # Route to appropriate handler
        handler_map = {
            'plant_info': lambda: self._plant_info_response(plants_found, user_input),
            'scientific_name': lambda: self._scientific_name_response(plants_found, user_input),
            'medicinal_uses': lambda: self._medicinal_uses_response(plants_found, user_input),
            'ethnobotanical_context': lambda: self._ethnobotanical_response(plants_found, user_input),
            'bioelectric_info': lambda: self._bioelectric_info_response(plants_found, user_input),
            'ailment_query': lambda: self._ailment_response(ailments_found),
            'stress_ailment': lambda: self._specific_ailment_response('stress'),
            'memory_ailment': lambda: self._specific_ailment_response('memory'),
            'respiratory_ailment': lambda: self._specific_ailment_response('respiratory'),
            'digestion_ailment': lambda: self._specific_ailment_response('digestion'),
            'immunity_ailment': lambda: self._specific_ailment_response('immunity'),
            'skin_ailment': lambda: self._specific_ailment_response('skin'),
            'inflammation_ailment': lambda: self._specific_ailment_response('inflammation'),
            'sleep_ailment': lambda: self._specific_ailment_response('sleep'),
            'nervine_query': lambda: self._nervine_response(),
            'bioelectric_general': lambda: self._bioelectric_general_response(),
            'list_plants': lambda: self._list_plants_response(),
            'compare_plants': lambda: self._compare_plants_response(plants_found),
        }

        if intent_tag in handler_map:
            return handler_map[intent_tag]()

        # For intents with static responses (greeting, goodbye, etc.)
        return self._static_response(intent_tag)

    def _static_response(self, intent_tag: str) -> str:
        """Get a random static response for the given intent."""
        for intent in self.intents_data['intents']:
            if intent['tag'] == intent_tag:
                responses = [r for r in intent['responses'] if not r.startswith('__')]
                if responses:
                    return random.choice(responses)
        return self._fallback_response({})

    def _fallback_response(self, entities: dict) -> str:
        """Generate a helpful fallback response."""
        plants = entities.get('plants', [])
        ailments = entities.get('ailments', [])

        if plants:
            return self._plant_info_response(plants, '')
        if ailments:
            return self._ailment_response(ailments)

        return ("I'm not entirely sure what you're asking, but I can help with:\n\n"
                "- **Plant Information**: 'Tell me about Ashwagandha'\n"
                "- **Ailment Remedies**: 'Which plants help with stress?'\n"
                "- **Bioelectric Properties**: 'Bioelectric properties of Brahmi'\n"
                "- **Nervine Tonics**: 'What are nervine tonics?'\n"
                "- **Traditional Uses**: 'Traditional use of Tulsi'\n"
                "- **All Plants**: 'List all plants'\n\n"
                "Try rephrasing your question, or ask about a specific plant!")

    def _get_plant(self, name: str) -> Optional[dict]:
        """Get plant data by name."""
        return self.plants.get(name.lower())

    def _plant_info_response(self, plant_names: list, user_input: str) -> str:
        """Full plant information response."""
        if not plant_names:
            return ("Which plant would you like to know about? I have information on: "
                    + ', '.join(p['name'] for p in self.plants_list[:10])
                    + f", and {len(self.plants_list) - 10} more.")

        plant = self._get_plant(plant_names[0])
        if not plant:
            return f"I don't have detailed information about '{plant_names[0]}' in my database yet."

        response = (
            f"**{plant['name']}** (*{plant['sciName']}*)\n"
            f"Family: {plant.get('family', 'N/A')}\n\n"
            f"{plant['description']}\n\n"
            f"**Medicinal Uses:**\n"
        )
        for use in plant['medicinalUses']:
            response += f"  - {use}\n"

        response += (
            f"\n**Bioelectric Properties:**\n{plant['bioelectric']}\n\n"
            f"**Ethnobotanical Context:**\n{plant['ethnobotanical']}\n\n"
        )

        if plant.get('activeCompounds'):
            response += f"**Active Compounds:** {', '.join(plant['activeCompounds'])}\n"
        if plant.get('partUsed'):
            response += f"**Parts Used:** {', '.join(plant['partUsed'])}\n"
        if plant.get('nervineType'):
            response += f"**Nervine Classification:** {plant['nervineType']}\n"

        response += "\nWould you like to know about another plant, or explore its properties in more detail?"
        return response

    def _scientific_name_response(self, plant_names: list, user_input: str) -> str:
        """Scientific name response."""
        if not plant_names:
            return "Which plant's scientific name would you like to know? Try asking: 'Scientific name of Tulsi'"

        plant = self._get_plant(plant_names[0])
        if not plant:
            return f"I don't have the scientific name for '{plant_names[0]}' in my database."

        response = (
            f"The scientific name of **{plant['name']}** is ***{plant['sciName']}***.\n\n"
            f"**Family:** {plant.get('family', 'N/A')}\n"
        )
        if plant.get('commonNames'):
            response += f"**Also known as:** {', '.join(plant['commonNames'])}\n"
        return response

    def _medicinal_uses_response(self, plant_names: list, user_input: str) -> str:
        """Medicinal uses response."""
        if not plant_names:
            return "Which plant's medicinal uses would you like to know? Try: 'Medicinal uses of Turmeric'"

        plant = self._get_plant(plant_names[0])
        if not plant:
            return f"I don't have medicinal information for '{plant_names[0]}'."

        response = f"**Medicinal Uses of {plant['name']}** (*{plant['sciName']}*):\n\n"
        for i, use in enumerate(plant['medicinalUses'], 1):
            response += f"{i}. {use}\n"

        if plant.get('activeCompounds'):
            response += f"\n**Key Active Compounds:** {', '.join(plant['activeCompounds'])}\n"
        if plant.get('partUsed'):
            response += f"**Parts Used:** {', '.join(plant['partUsed'])}\n"

        response += ("\n*Disclaimer: This information is for educational purposes only. "
                    "Always consult a healthcare professional before use.*")
        return response

    def _ethnobotanical_response(self, plant_names: list, user_input: str) -> str:
        """Ethnobotanical context response."""
        if not plant_names:
            return "Which plant's traditional context would you like to explore? Try: 'Traditional use of Neem'"

        plant = self._get_plant(plant_names[0])
        if not plant:
            return f"I don't have ethnobotanical data for '{plant_names[0]}'."

        response = (
            f"**Ethnobotanical Context of {plant['name']}** (*{plant['sciName']}*):\n\n"
            f"{plant['ethnobotanical']}\n\n"
        )
        if plant.get('habitat'):
            response += f"**Natural Habitat:** {plant['habitat']}\n"
        if plant.get('partUsed'):
            response += f"**Traditionally Used Parts:** {', '.join(plant['partUsed'])}\n"

        return response

    def _bioelectric_info_response(self, plant_names: list, user_input: str) -> str:
        """Bioelectric properties response for specific plants."""
        if not plant_names:
            return ("Which plant's bioelectric properties interest you? "
                    "Plants with notable bioelectric effects include: "
                    + ', '.join(p['name'] for p in self.nervine_plants)
                    + ". Try: 'Bioelectric properties of Brahmi'")

        plant = self._get_plant(plant_names[0])
        if not plant:
            return f"I don't have bioelectric data for '{plant_names[0]}'."

        response = (
            f"**Bioelectric Properties of {plant['name']}** (*{plant['sciName']}*):\n\n"
            f"{plant['bioelectric']}\n\n"
        )
        if plant.get('nervineType'):
            response += f"**Nervine Classification:** {plant['nervineType']}\n"
        if plant.get('activeCompounds'):
            response += f"**Key Compounds Involved:** {', '.join(plant['activeCompounds'])}\n"

        response += ("\nThis demonstrates the bridge between traditional healing "
                    "descriptions and modern bioelectronics understanding.")
        return response

    def _ailment_response(self, ailments: list) -> str:
        """Response for ailment-based queries."""
        if not ailments:
            return ("What health concern are you looking for plants for? I can help with: "
                    "stress, memory, respiratory issues, digestion, immunity, skin, "
                    "inflammation, and sleep.")

        ailment = ailments[0]
        ailment_labels = {
            'stress': 'Stress & Anxiety',
            'memory': 'Cognitive Health & Memory',
            'respiratory': 'Respiratory Health',
            'digestion': 'Digestive Wellness',
            'immunity': 'Immunity Support',
            'skin': 'Skin Health',
            'inflammation': 'Anti-Inflammatory',
            'sleep': 'Sleep & Relaxation'
        }

        plants = self.ailment_plant_map.get(ailment, [])
        if not plants:
            return f"I don't have specific plant recommendations for '{ailment}' in my database."

        label = ailment_labels.get(ailment, ailment.title())
        response = f"**Plants for {label}** in traditional Indian ethnobotany:\n\n"

        for i, plant in enumerate(plants, 1):
            # Get the most relevant medicinal use
            relevant_use = plant['medicinalUses'][0] if plant['medicinalUses'] else 'General wellness'
            response += f"{i}. **{plant['name']}** (*{plant['sciName']}*) - {relevant_use}\n"

        response += ("\n*Always consult a healthcare professional before using any plant-based remedy.* "
                    "Would you like detailed information on any of these plants?")
        return response

    def _specific_ailment_response(self, ailment: str) -> str:
        """Response for specific ailment intent tags."""
        return self._ailment_response([ailment])

    def _nervine_response(self) -> str:
        """Response about nervine tonics."""
        response = (
            "**Nervine Tonics in Indian Ethnobotany**\n\n"
            "Nervine tonics are plants that support and strengthen the nervous system. "
            "They bridge ancient Ayurvedic wisdom with modern bioelectronics by modulating "
            "ion channels, neurotransmitter activity, and neural signaling patterns.\n\n"
            "Key nervine plants in our database:\n\n"
        )

        for i, plant in enumerate(self.nervine_plants, 1):
            response += (
                f"{i}. **{plant['name']}** (*{plant['sciName']}*)\n"
                f"   Classification: {plant.get('nervineType', 'N/A')}\n"
                f"   Mechanism: {plant['bioelectric'].split('.')[0]}.\n\n"
            )

        response += ("These plants demonstrate how traditional descriptions like 'calming the nerves' "
                     "connect to measurable bioelectric phenomena in modern neuroscience.")
        return response

    def _bioelectric_general_response(self) -> str:
        """General bioelectric properties response."""
        return self._static_response('bioelectric_general')

    def _list_plants_response(self) -> str:
        """List all plants in the database."""
        response = f"**Complete Plant Database** ({len(self.plants_list)} plants):\n\n"

        for i, plant in enumerate(self.plants_list, 1):
            ailment_tags = ', '.join(plant.get('ailments', []))
            response += f"{i}. **{plant['name']}** (*{plant['sciName']}*) - {ailment_tags}\n"

        response += "\nAsk me about any specific plant for detailed information!"
        return response

    def _compare_plants_response(self, plant_names: list) -> str:
        """Compare two plants."""
        if len(plant_names) < 2:
            return ("To compare plants, mention two plant names. For example: "
                    "'Compare Brahmi and Ashwagandha' or 'Difference between Tulsi and Neem'")

        p1 = self._get_plant(plant_names[0])
        p2 = self._get_plant(plant_names[1])

        if not p1 or not p2:
            missing = plant_names[0] if not p1 else plant_names[1]
            return f"I don't have '{missing}' in my database for comparison."

        response = f"**Comparison: {p1['name']} vs {p2['name']}**\n\n"
        response += f"| Aspect | {p1['name']} | {p2['name']} |\n"
        response += f"|--------|{'---' * 5}|{'---' * 5}|\n"
        response += f"| Scientific Name | *{p1['sciName']}* | *{p2['sciName']}* |\n"
        response += f"| Family | {p1.get('family', 'N/A')} | {p2.get('family', 'N/A')} |\n"
        response += f"| Key Uses | {', '.join(p1['medicinalUses'][:3])} | {', '.join(p2['medicinalUses'][:3])} |\n"
        response += f"| Nervine Type | {p1.get('nervineType') or 'N/A'} | {p2.get('nervineType') or 'N/A'} |\n"
        response += f"| Parts Used | {', '.join(p1.get('partUsed', ['N/A']))} | {', '.join(p2.get('partUsed', ['N/A']))} |\n"

        response += f"\n**{p1['name']} Bioelectric:** {p1['bioelectric'].split('.')[0]}.\n"
        response += f"\n**{p2['name']} Bioelectric:** {p2['bioelectric'].split('.')[0]}.\n"

        return response
