

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# --- Medical database ---
MEDICAL_DB = {
  "flu": {
    "symptoms": ["Fever", "Cough and colds", "Body pain", "Headache", "Throat itchiness"],
    "medications": ["Paracetamol", "Rest", "Hydration"],
    "advice": "Stay home, rest, and drink plenty of fluids.",
    "risk": "Moderate",
    "severity": "Mild",
  },
  "chills": {
    "symptoms": ["Chills", "Fever", "Body Weakness"],
    "medications": ["Paracetamol", "Warm fluids", "Rest"],
    "advice": "Keep warm, monitor temperature, and drink plenty of fluids.",
    "risk": "Low",
    "severity": "Mild",
  },
  "sore_throat": {
    "symptoms": ["Throat itchiness", "Cough and colds", "Fever"],
    "medications": ["Lozenges", "Salt water gargle", "Paracetamol"],
    "advice": "Avoid cold drinks and rest the voice.",
    "risk": "Low",
    "severity": "Mild",
  },
  "respiratory_distress": {
    "symptoms": ["Difficulty of Breathing", "Chest heaviness", "Hyperventilation"],
    "medications": ["Oxygen therapy", "Bronchodilators", "Nebulization"],
    "advice": "Seek immediate medical attention.",
    "risk": "High",
    "severity": "Severe",
  },
  "dysphagia": {
    "symptoms": ["Dysphagia", "Throat itchiness", "Cough and colds"],
    "medications": ["Soft diet", "Warm fluids", "Medical consultation"],
    "advice": "Eat slowly and consult a doctor if persistent.",
    "risk": "Moderate",
    "severity": "Moderate",
  },
  "migraine": {
    "symptoms": ["Headache", "Nape Pain", "Dizziness"],
    "medications": ["Ibuprofen", "Caffeine", "Cold compress"],
    "advice": "Stay in a quiet, dark room. Avoid known triggers.",
    "risk": "Low",
    "severity": "Moderate",
  },
  "unconsciousness": {
    "symptoms": ["LOC", "Dizziness", "Vehucular Accident"],
    "medications": ["Oxygen", "Hospital observation", "IV fluids"],
    "advice": "Call emergency services immediately.",
    "risk": "High",
    "severity": "Severe",
  },
  "eye_allergy": {
    "symptoms": ["Eye Itchiness/redness", "Swelling eye lid"],
    "medications": ["Antihistamine drops", "Cold compress", "Avoid allergens"],
    "advice": "Do not rub eyes. Rinse with clean water if irritated.",
    "risk": "Low",
    "severity": "Mild",
  },
  "ear_infection": {
    "symptoms": ["Ear pain", "Fever", "Headache"],
    "medications": ["Antibiotics", "Paracetamol", "Ear drops"],
    "advice": "Avoid inserting objects into the ear. Seek medical attention.",
    "risk": "Moderate",
    "severity": "Moderate",
  },
  "chest_pain": {
    "symptoms": ["Chest heaviness", "Palpitation", "Hyperventilation"],
    "medications": ["Aspirin", "Oxygen", "ECG evaluation"],
    "advice": "Rest and seek emergency care immediately.",
    "risk": "High",
    "severity": "Severe",
  },
  "toothache": {
    "symptoms": ["Toothache", "Headache", "Dizziness"],
    "medications": ["Mefenamic acid", "Saltwater rinse", "Dental consultation"],
    "advice": "Avoid cold or sweet foods. Visit a dentist.",
    "risk": "Low",
    "severity": "Mild",
  },
  "dizziness": {
    "symptoms": ["Dizziness", "Nausea and vomiting", "LOC"],
    "medications": ["Meclizine", "Hydration", "Rest"],
    "advice": "Avoid sudden movements. Sit or lie down immediately if dizzy.",
    "risk": "Moderate",
    "severity": "Mild",
  },
  "gastritis": {
    "symptoms": ["Epigastric pain", "Nausea and vomiting", "Abdominal pain/RLQ Pain/LLQ"],
    "medications": ["Antacids", "Omeprazole", "Diet modification"],
    "advice": "Avoid acidic foods. Eat small, frequent meals.",
    "risk": "Low",
    "severity": "Mild",
  },
  "abdominal_pain": {
    "symptoms": ["Abdominal pain/RLQ Pain/LLQ", "Fever", "Nausea and vomiting"],
    "medications": ["Buscopan", "IV fluids", "Observation"],
    "advice": "Avoid solid food. Seek medical care if persistent.",
    "risk": "Moderate",
    "severity": "Varies",
  },
  "uti": {
    "symptoms": ["Hypogastric pain", "Dysuria", "Body Weakness"],
    "medications": ["Antibiotics", "Cranberry juice", "Pain relievers"],
    "advice": "Increase fluid intake. Complete antibiotic course.",
    "risk": "Moderate",
    "severity": "Moderate",
  },
  "lymphadenitis": {
    "symptoms": ["Swelling armpit", "Fever", "Body pain"],
    "medications": ["Antibiotics", "Warm compress", "Pain relievers"],
    "advice": "Monitor for fever. Seek care for persistent swelling.",
    "risk": "Moderate",
    "severity": "Mild",
  },
  "palpitations": {
    "symptoms": ["Palpitation", "Dizziness", "Chest heaviness"],
    "medications": ["Beta blockers", "Electrolyte balancing", "ECG monitoring"],
    "advice": "Avoid caffeine and stress. Seek cardiac evaluation.",
    "risk": "Moderate",
    "severity": "Moderate"
  },
  "muscle_pain": {
    "symptoms": ["Shoulder Pain", "Body pain", "Nape Pain"],
    "medications": ["Pain relievers", "Massage", "Hot compress"],
    "advice": "Rest affected area. Avoid strenuous activity.",
    "risk": "Low",
    "severity": "Mild",
  },
  "diarrhea": {
    "symptoms": ["LBM/diarrhea", "Abdominal pain/RLQ Pain/LLQ", "Body Weakness"],
    "medications": ["ORS", "Loperamide", "Zinc supplements"],
    "advice": "Rehydrate regularly. Avoid greasy food.",
    "risk": "Moderate",
    "severity": "Mild",
  },
  "constipation": {
    "symptoms": ["Constipation", "Abdominal pain/RLQ Pain/LLQ", "Hypogastric pain"],
    "medications": ["Laxatives", "High-fiber diet", "Water"],
    "advice": "Increase fiber intake and exercise regularly.",
    "risk": "Low",
    "severity": "Mild",
  },
  "pregnancy": {
    "symptoms": ["Pregnancy", "Nausea and vomiting", "Abdominal pain/RLQ Pain/LLQ"],
    "medications": ["Prenatal vitamins", "Iron", "Folic acid"],
    "advice": "Attend prenatal check-ups regularly.",
    "risk": "Low",
    "severity": "Normal",
  },
  "fatigue": {
    "symptoms": ["Body Weakness", "Dizziness", "Palpitation"],
    "medications": ["Vitamins", "Rest", "Iron supplements"],
    "advice": "Get enough sleep and eat nutritious food.",
    "risk": "Low",
    "severity": "Mild",
  },
  "vehicular_accident": {
    "symptoms": ["Vehucular Accident", "LOC", "Laceration"],
    "medications": ["Wound care", "CT scan", "Pain relievers"],
    "advice": "Do not move the injured. Call emergency services immediately.",
    "risk": "High",
    "severity": "Severe",
  },
  "rashes": {
    "symptoms": ["Rashes", "Skin itchiness", "Body Weakness"],
    "medications": ["Antihistamines", "Hydrocortisone cream", "Calamine lotion"],
    "advice": "Avoid irritants. Do not scratch the area.",
    "risk": "Low",
    "severity": "Mild"
  },
  "cat_bite": {
    "symptoms": ["Cat Bite", "Swelling", "Pain"],
    "medications": ["Anti-rabies vaccine", "Antibiotics", "Wound care"],
    "advice": "Clean the bite with soap and water. Get vaccinated promptly.",
    "risk": "High",
    "severity": "Moderate",
  },
  "foot_pain": {
    "symptoms": ["Foot pain", "Swelling", "Body Weakness"],
    "medications": ["Pain relievers", "Rest", "Cold compress"],
    "advice": "Avoid standing for long. Use proper footwear.",
    "risk": "Low",
    "severity": "Mild",
  },
  "laceration": {
    "symptoms": ["Laceration", "Bleeding", "Swelling"],
    "medications": ["Antiseptic", "Wound dressing", "Pain relievers"],
    "advice": "Apply pressure to stop bleeding. Keep wound clean.",
    "risk": "Moderate",
    "severity": "Moderate",
  },
  "punctured_wound": {
    "symptoms": ["Punctured wound", "Pain", "Swelling"],
    "medications": ["Tetanus shot", "Antibiotics", "Wound cleaning"],
    "advice": "Wash with soap and water. Do not remove embedded objects.",
    "risk": "High",
    "severity": "Moderate",
  },
  "burn": {
    "symptoms": ["Burn", "Pain", "Swelling"],
    "medications": ["Cold water", "Burn ointment", "Paracetamol"],
    "advice": "Cool the area. Do not apply ice or toothpaste.",
    "risk": "Moderate",
    "severity": "Varies",
  }
}


def create_training_data():
    diseases = []
    symptoms_list = []
    for disease, info in MEDICAL_DB.items():
        diseases.extend([disease] * 100)
        symptoms = info["symptoms"]
        for _ in range(100):
            present = np.random.choice(symptoms, size=np.random.randint(2, len(symptoms)+1), replace=False)
            symptoms_list.append(list(present))
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(symptoms_list)
    le = LabelEncoder()
    y = le.fit_transform(diseases)
    df = pd.DataFrame(X, columns=mlb.classes_)
    df['disease'] = y
    return df, mlb, le

app = Flask(__name__)
CORS(app)

df, mlb, le = create_training_data()
model = RandomForestClassifier(n_estimators=200)
model.fit(df.drop('disease', axis=1), df['disease'])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms', [])
    input_vec = [1 if s in symptoms else 0 for s in mlb.classes_]
    proba = model.predict_proba([input_vec])[0]
    disease_indices = model.classes_
    disease_names = le.inverse_transform(disease_indices)
    results = sorted(zip(disease_names, proba), key=lambda x: -x[1])
    top_results = []
    for disease, prob in results[:3]:
        if prob < 0.1:
            continue
        info = MEDICAL_DB.get(disease, {})
        top_results.append({
            'disease': disease,
            'probability': float(prob),
            'info': info
        })
    return jsonify({'predictions': top_results})

if __name__ == '__main__':
    app.run(port=5000, debug=True)