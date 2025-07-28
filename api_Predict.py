from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from collections import defaultdict
import re

# --- Medical database ---
MEDICAL_DB = {
  "flu": {
    "symptoms": ["fever", "cough_and_colds", "body_pain", "headache", "throat_itchiness"],
    "medications": ["Paracetamol", "Rest", "Hydration"],
    "advice": "Stay home, rest, and drink plenty of fluids.",
    "risk": "Moderate",
    "severity": "Mild"
  },
  "chills": {
    "symptoms": ["chills", "fever", "body_weakness"],
    "medications": ["Paracetamol", "Warm fluids", "Rest"],
    "advice": "Keep warm, monitor temperature, and drink plenty of fluids.",
    "risk": "Low",
    "severity": "Mild"
  },
  "sore_throat": {
    "symptoms": ["throat_itchiness", "cough_and_colds", "fever"],
    "medications": ["Lozenges", "Salt water gargle", "Paracetamol"],
    "advice": "Avoid cold drinks and rest the voice.",
    "risk": "Low",
    "severity": "Mild"
  },
  "respiratory_distress": {
    "symptoms": ["difficulty_of_breathing", "chest_heaviness", "hyperventilation"],
    "medications": ["Oxygen therapy", "Bronchodilators", "Nebulization"],
    "advice": "Seek immediate medical attention.",
    "risk": "High",
    "severity": "Severe"
  },
  "dysphagia": {
    "symptoms": ["dysphagia", "throat_itchiness", "cough_and_colds"],
    "medications": ["Soft diet", "Warm fluids", "Medical consultation"],
    "advice": "Eat slowly and consult a doctor if persistent.",
    "risk": "Moderate",
    "severity": "Moderate"
  },
  "migraine": {
    "symptoms": ["headache", "nape_pain", "dizziness"],
    "medications": ["Ibuprofen", "Caffeine", "Cold compress"],
    "advice": "Stay in a quiet, dark room. Avoid known triggers.",
    "risk": "Low",
    "severity": "Moderate"
  },
  "unconsciousness": {
    "symptoms": ["loc", "dizziness", "vehicular_accident"],
    "medications": ["Oxygen", "Hospital observation", "IV fluids"],
    "advice": "Call emergency services immediately.",
    "risk": "High",
    "severity": "Severe"
  },
  "eye_allergy": {
    "symptoms": ["eye_itchiness_redness", "swelling_eye_lid"],
    "medications": ["Antihistamine drops", "Cold compress", "Avoid allergens"],
    "advice": "Do not rub eyes. Rinse with clean water if irritated.",
    "risk": "Low",
    "severity": "Mild"
  },
  "ear_infection": {
    "symptoms": ["ear_pain", "fever", "headache"],
    "medications": ["Antibiotics", "Paracetamol", "Ear drops"],
    "advice": "Avoid inserting objects into the ear. Seek medical attention.",
    "risk": "Moderate",
    "severity": "Moderate"
  },
  "chest_pain": {
    "symptoms": ["chest_heaviness", "palpitation", "hyperventilation"],
    "medications": ["Aspirin", "Oxygen", "ECG evaluation"],
    "advice": "Rest and seek emergency care immediately.",
    "risk": "High",
    "severity": "Severe"
  },
  "toothache": {
    "symptoms": ["toothache", "headache", "dizziness"],
    "medications": ["Mefenamic acid", "Saltwater rinse", "Dental consultation"],
    "advice": "Avoid cold or sweet foods. Visit a dentist.",
    "risk": "Low",
    "severity": "Mild"
  },
  "vertigo": {
    "symptoms": ["dizziness", "nausea_and_vomiting", "loc"],
    "medications": ["Meclizine", "Hydration", "Rest"],
    "advice": "Avoid sudden movements. Sit or lie down immediately if dizzy.",
    "risk": "Moderate",
    "severity": "Mild"
  },
  "gastritis": {
    "symptoms": ["epigastric_pain", "nausea_and_vomiting", "abdominal_pain"],
    "medications": ["Antacids", "Omeprazole", "Diet modification"],
    "advice": "Avoid acidic foods. Eat small, frequent meals.",
    "risk": "Low",
    "severity": "Mild"
  },
  "abdominal_pain": {
    "symptoms": ["abdominal_pain", "fever", "nausea_and_vomiting"],
    "medications": ["Buscopan", "IV fluids", "Observation"],
    "advice": "Avoid solid food. Seek medical care if persistent.",
    "risk": "Moderate",
    "severity": "Varies"
  },
  "uti": {
    "symptoms": ["hypogastric_pain", "dysuria", "body_weakness"],
    "medications": ["Antibiotics", "Cranberry juice", "Pain relievers"],
    "advice": "Increase fluid intake. Complete antibiotic course.",
    "risk": "Moderate",
    "severity": "Moderate"
  },
  "lymphadenitis": {
    "symptoms": ["swelling_armpit", "fever", "body_pain"],
    "medications": ["Antibiotics", "Warm compress", "Pain relievers"],
    "advice": "Monitor for fever. Seek care for persistent swelling.",
    "risk": "Moderate",
    "severity": "Mild"
  },
  "palpitations": {
    "symptoms": ["palpitation", "dizziness", "chest_heaviness"],
    "medications": ["Beta blockers", "Electrolyte balancing", "ECG monitoring"],
    "advice": "Avoid caffeine and stress. Seek cardiac evaluation.",
    "risk": "Moderate",
    "severity": "Moderate"
  },
  "muscle_pain": {
    "symptoms": ["shoulder_pain", "body_pain", "nape_pain"],
    "medications": ["Pain relievers", "Massage", "Hot compress"],
    "advice": "Rest affected area. Avoid strenuous activity.",
    "risk": "Low",
    "severity": "Mild"
  },
  "diarrhea": {
    "symptoms": ["lbm_diarrhea", "abdominal_pain", "body_weakness"],
    "medications": ["ORS", "Loperamide", "Zinc supplements"],
    "advice": "Rehydrate regularly. Avoid greasy food.",
    "risk": "Moderate",
    "severity": "Mild"
  },
  "constipation": {
    "symptoms": ["constipation", "abdominal_pain", "hypogastric_pain"],
    "medications": ["Laxatives", "High-fiber diet", "Water"],
    "advice": "Increase fiber intake and exercise regularly.",
    "risk": "Low",
    "severity": "Mild"
  },
  "pregnancy_symptoms": {
    "symptoms": ["pregnancy", "nausea_and_vomiting", "abdominal_pain"],
    "medications": ["Prenatal vitamins", "Iron", "Folic acid"],
    "advice": "Attend prenatal check-ups regularly.",
    "risk": "Low",
    "severity": "Normal"
  },
  "fatigue": {
    "symptoms": ["body_weakness", "dizziness", "palpitation"],
    "medications": ["Vitamins", "Rest", "Iron supplements"],
    "advice": "Get enough sleep and eat nutritious food.",
    "risk": "Low",
    "severity": "Mild"
  },
  "trauma": {
    "symptoms": ["vehicular_accident", "loc", "laceration"],
    "medications": ["Wound care", "CT scan", "Pain relievers"],
    "advice": "Do not move the injured. Call emergency services immediately.",
    "risk": "High",
    "severity": "Severe"
  },
  "skin_allergy": {
    "symptoms": ["rashes", "skin_itchiness", "body_weakness"],
    "medications": ["Antihistamines", "Hydrocortisone cream", "Calamine lotion"],
    "advice": "Avoid irritants. Do not scratch the area.",
    "risk": "Low",
    "severity": "Mild"
  },
  "animal_bite": {
    "symptoms": ["cat_bite", "rashes", "body_pain"],
    "medications": ["Anti-rabies vaccine", "Antibiotics", "Wound care"],
    "advice": "Clean the bite with soap and water. Get vaccinated promptly.",
    "risk": "High",
    "severity": "Moderate"
  },
  "foot_injury": {
    "symptoms": ["foot_pain", "body_weakness", "body_pain"],
    "medications": ["Pain relievers", "Rest", "Cold compress"],
    "advice": "Avoid standing for long. Use proper footwear.",
    "risk": "Low",
    "severity": "Mild"
  },
  "wound": {
    "symptoms": ["laceration", "body_pain", "punctured_wound"],
    "medications": ["Antiseptic", "Wound dressing", "Pain relievers"],
    "advice": "Apply pressure to stop bleeding. Keep wound clean.",
    "risk": "Moderate",
    "severity": "Moderate"
  },
  "puncture_injury": {
    "symptoms": ["punctured_wound", "body_pain", "foot_pain"],
    "medications": ["Tetanus shot", "Antibiotics", "Wound cleaning"],
    "advice": "Wash with soap and water. Do not remove embedded objects.",
    "risk": "High",
    "severity": "Moderate"
  },
  "burn_injury": {
    "symptoms": ["burn", "body_pain", "rashes"],
    "medications": ["Cold water", "Burn ointment", "Paracetamol"],
    "advice": "Cool the area. Do not apply ice or toothpaste.",
    "risk": "Moderate",
    "severity": "Varies"
  }
}

# All possible symptoms from your frontend
ALL_SYMPTOMS = [
    'fever', 'chills', 'cough_and_colds', 'throat_itchiness', 'difficulty_of_breathing',
    'dysphagia', 'headache', 'loc', 'eye_itchiness_redness', 'swelling_eye_lid',
    'ear_pain', 'chest_heaviness', 'hyperventilation', 'toothache', 'dizziness',
    'nausea_and_vomiting', 'epigastric_pain', 'abdominal_pain', 'hypogastric_pain',
    'dysuria', 'swelling_armpit', 'body_pain', 'palpitation', 'shoulder_pain',
    'lbm_diarrhea', 'constipation', 'pregnancy', 'body_weakness', 'nape_pain',
    'vehicular_accident', 'rashes', 'skin_itchiness', 'cat_bite', 'foot_pain',
    'laceration', 'punctured_wound', 'burn'
]

def normalize_symptom(symptom):
    """Normalize symptom names to match database format"""
    if not symptom:
        return ""
    
    # Convert to lowercase and replace spaces/special chars with underscores
    normalized = re.sub(r'[^\w\s]', '', str(symptom).lower())
    normalized = re.sub(r'\s+', '_', normalized.strip())
    
    # Handle special cases and variations
    mappings = {
        'cough_and_cold': 'cough_and_colds',
        'cough': 'cough_and_colds',
        'cold': 'cough_and_colds',
        'sore_throat': 'throat_itchiness',
        'loss_of_consciousness': 'loc',
        'unconscious': 'loc',
        'eye_redness': 'eye_itchiness_redness',
        'eye_irritation': 'eye_itchiness_redness',
        'swollen_eyelid': 'swelling_eye_lid',
        'chest_pain': 'chest_heaviness',
        'stomach_pain': 'abdominal_pain',
        'belly_pain': 'abdominal_pain',
        'heart_palpitation': 'palpitation',
        'racing_heart': 'palpitation',
        'loose_bowel_movement': 'lbm_diarrhea',
        'diarrhea': 'lbm_diarrhea',
        'lbm': 'lbm_diarrhea',
        'neck_pain': 'nape_pain',
        'car_accident': 'vehicular_accident',
        'accident': 'vehicular_accident',
        'skin_rash': 'rashes',
        'rash': 'rashes',
        'itchy_skin': 'skin_itchiness',
        'animal_bite': 'cat_bite',
        'bite': 'cat_bite',
        'cut': 'laceration',
        'wound': 'laceration',
        'puncture': 'punctured_wound',
        'stab_wound': 'punctured_wound'
    }
    
    return mappings.get(normalized, normalized)

def create_training_data():
    """Create comprehensive training data with better symptom combinations"""
    diseases = []
    symptoms_list = []
    
    for disease, info in MEDICAL_DB.items():
        primary_symptoms = info["symptoms"]
        
        # Generate more realistic training examples
        for _ in range(150):  # More training examples per disease
            
            # Always include at least one primary symptom (higher probability for first symptoms)
            selected_symptoms = []
            
            # Primary symptoms (disease-specific) - higher probability
            for i, symptom in enumerate(primary_symptoms):
                # First symptom has 90% chance, others decrease
                prob = max(0.9 - (i * 0.15), 0.3)
                if np.random.random() < prob:
                    selected_symptoms.append(symptom)
            
            # Ensure at least one primary symptom is always present
            if not selected_symptoms:
                selected_symptoms.append(np.random.choice(primary_symptoms))
            
            # Add some related symptoms from other conditions (noise) - lower probability
            other_symptoms = [s for disease_info in MEDICAL_DB.values() 
                            for s in disease_info["symptoms"] 
                            if s not in primary_symptoms]
            
            # Add 0-2 random symptoms with low probability
            num_noise = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
            if num_noise > 0 and other_symptoms:
                noise_symptoms = np.random.choice(other_symptoms, 
                                                size=min(num_noise, len(other_symptoms)), 
                                                replace=False)
                selected_symptoms.extend(noise_symptoms)
            
            symptoms_list.append(selected_symptoms)
            diseases.append(disease)
    
    # Create feature matrix
    feature_matrix = []
    for symptoms in symptoms_list:
        feature_vector = [1 if symptom in symptoms else 0 for symptom in ALL_SYMPTOMS]
        feature_matrix.append(feature_vector)
    
    X = np.array(feature_matrix)
    
    # Encode diseases
    le = LabelEncoder()
    y = le.fit_transform(diseases)
    
    return X, y, le

def calculate_confidence_score(probabilities, selected_symptoms, disease_info):
    """Calculate a more realistic confidence score"""
    base_prob = probabilities[0] if len(probabilities) > 0 else 0
    
    if not disease_info or 'symptoms' not in disease_info:
        return base_prob * 0.5  # Low confidence for unknown diseases
    
    disease_symptoms = set(disease_info['symptoms'])
    user_symptoms = set(selected_symptoms)
    
    # Calculate symptom match ratio
    if len(disease_symptoms) == 0:
        match_ratio = 0
    else:
        matches = len(disease_symptoms.intersection(user_symptoms))
        match_ratio = matches / len(disease_symptoms)
    
    # Adjust confidence based on symptom matching
    if match_ratio >= 0.8:  # 80% or more symptoms match
        confidence = min(base_prob * 1.2, 0.95)
    elif match_ratio >= 0.6:  # 60-79% match
        confidence = base_prob * 1.1
    elif match_ratio >= 0.4:  # 40-59% match
        confidence = base_prob
    elif match_ratio >= 0.2:  # 20-39% match
        confidence = base_prob * 0.8
    else:  # Less than 20% match
        confidence = base_prob * 0.5
    
    return max(min(confidence, 0.95), 0.05)  # Keep between 5% and 95%

app = Flask(__name__)
CORS(app)

# Initialize model
print("Creating training data...")
X, y, le = create_training_data()

print("Training model...")
model = RandomForestClassifier(
    n_estimators=300,  # More trees for better accuracy
    max_depth=15,      # Prevent overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X, y)
print("Model training completed!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        personal = data.get('personal', {})
        
        if not symptoms:
            return jsonify({
                'predictions': [],
                'message': 'No symptoms provided'
            })
        
        # Normalize and filter symptoms
        normalized_symptoms = []
        for symptom in symptoms:
            if isinstance(symptom, str) and symptom.strip():
                normalized = normalize_symptom(symptom.strip())
                if normalized in ALL_SYMPTOMS:
                    normalized_symptoms.append(normalized)
                else:
                    # Handle custom symptoms by trying to map them
                    print(f"Unknown symptom: {symptom} -> {normalized}")
        
        if not normalized_symptoms:
            return jsonify({
                'predictions': [{
                    'disease': 'unknown_condition',
                    'probability': 0.1,
                    'confidence': 10,
                    'info': {
                        'symptoms': symptoms,
                        'medications': ['Consult healthcare provider'],
                        'advice': 'Please consult with a healthcare professional for proper diagnosis.',
                        'risk': 'Unknown',
                        'severity': 'Unknown'
                    }
                }],
                'message': 'Unable to recognize the provided symptoms'
            })
        
        # Create input vector
        input_vector = [1 if symptom in normalized_symptoms else 0 for symptom in ALL_SYMPTOMS]
        
        # Get predictions
        probabilities = model.predict_proba([input_vector])[0]
        disease_indices = model.classes_
        disease_names = le.inverse_transform(disease_indices)
        
        # Sort results by probability
        results = list(zip(disease_names, probabilities))
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Format top results
        top_results = []
        for disease, prob in results[:5]:  # Top 5 predictions
            if prob < 0.05:  # Skip very low probability predictions
                continue
                
            disease_info = MEDICAL_DB.get(disease, {})
            confidence = calculate_confidence_score([prob], normalized_symptoms, disease_info)
            
            # Format the result
            result = {
                'disease': disease.replace('_', ' ').title(),
                'probability': float(prob),
                'confidence': round(confidence * 100, 1),
                'info': {
                    'symptoms': disease_info.get('symptoms', []),
                    'medications': disease_info.get('medications', ['Consult healthcare provider']),
                    'advice': disease_info.get('advice', 'Please consult with a healthcare professional.'),
                    'risk': disease_info.get('risk', 'Unknown'),
                    'severity': disease_info.get('severity', 'Unknown')
                }
            }
            top_results.append(result)
        
        # If no good predictions, provide a general result
        if not top_results:
            top_results = [{
                'disease': 'General Symptoms',
                'probability': 0.3,
                'confidence': 30,
                'info': {
                    'symptoms': normalized_symptoms,
                    'medications': ['Rest', 'Hydration', 'Monitor symptoms'],
                    'advice': 'Monitor symptoms closely. Consult healthcare provider if symptoms persist or worsen.',
                    'risk': 'Low to Moderate',
                    'severity': 'Mild'
                }
            }]
        
        return jsonify({
            'predictions': top_results,
            'input_symptoms': normalized_symptoms,
            'message': 'Prediction completed successfully'
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'predictions': []
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_trained': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)