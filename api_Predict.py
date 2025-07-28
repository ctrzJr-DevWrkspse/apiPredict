

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# --- Medical database ---
MEDICAL_DB = {
    "flu": {
        "symptoms": ["fever", "cough", "fatigue", "headache", "sore_throat"],
        "medications": ["Paracetamol", "Rest", "Hydration"],
        "advice": "Stay home, rest, and drink plenty of fluids.",
        "risk": "Moderate",
        "severity": "Mild"
    },
    "covid-19": {
        "symptoms": ["fever", "cough", "shortness_of_breath", "loss_of_taste", "fatigue"],
        "medications": ["Paracetamol", "Antivirals", "Monitoring oxygen levels"],
        "advice": "Isolate, monitor symptoms, seek medical attention if worsening.",
        "risk": "High",
        "severity": "Severe"
    },
    "cold": {
        "symptoms": ["sore_throat", "runny_nose", "sneezing", "cough", "congestion"],
        "medications": ["Antihistamines", "Decongestants", "Rest"],
        "advice": "Drink fluids and use over-the-counter cold remedies.",
        "risk": "Low",
        "severity": "Mild"
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