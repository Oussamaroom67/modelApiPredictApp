from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

model = joblib.load("predictModel.joblib")
symptom_weights = pd.read_csv("Symptom-severity-adjusted.csv")
description_df = pd.read_csv("symptom_Description.csv")
precaution_df = pd.read_csv("symptom_precaution.csv")

app = Flask(__name__)

def encode_symptoms(symptoms):
    symptoms_encoded = []
    for symptom in symptoms:
        if symptom in symptom_weights['Symptom'].values:
            weight = symptom_weights.loc[symptom_weights['Symptom'] == symptom, 'weight'].values[0]
            symptoms_encoded.append(weight)
        else:
            symptoms_encoded.append(0)  
    while len(symptoms_encoded) < 17:
        symptoms_encoded.append(0)
    return np.array(symptoms_encoded).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Requête reçue")
        data = request.get_json()
        print("Données reçues :", data)

        symptoms = data.get("symptoms", [])
        
        if not symptoms:
            return jsonify({"error": "Aucun symptôme fourni"}), 400
        
        processed_symptoms = [symptom.replace(' ', '_') for symptom in symptoms]
        print("Symptômes reçus :", processed_symptoms)

        # Encodage des symptômes
        encoded_symptoms = encode_symptoms(processed_symptoms)
        print("Symptômes encodés :", encoded_symptoms)

        # Prédiction
        probabilities = model.predict_proba(encoded_symptoms)[0]
        prediction = model.predict(encoded_symptoms)[0]
        predicted_probability = probabilities[model.classes_.tolist().index(prediction)]
        print("Maladie prédite :", prediction)
        print("Probabilité associée :", predicted_probability)

        description = description_df.loc[description_df['Disease'] == prediction, 'Description'].values[0]
        precautions = precaution_df.loc[precaution_df['Disease'] == prediction].iloc[0, 1:].dropna().tolist()
        print("Description :", description)
        print("Précautions :", precautions)
        
        return jsonify({
            "disease": prediction,
            "confidence": predicted_probability,
            "description": description,
            "precautions": precautions
        })
    except Exception as e:
        print("Erreur :", e)  
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
