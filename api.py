from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Chargement des fichiers et du modèle
model = joblib.load("medicalModel.joblib")
symptom_weights = pd.read_csv("Symptom-severity-adjusted.csv")
description_df = pd.read_csv("symptom_Description.csv")
precaution_df = pd.read_csv("symptom_precaution.csv")

# Initialisation de l'application Flask
app = Flask(__name__)

# Fonction pour encoder les symptômes
def encode_symptoms(symptoms):
    symptoms_encoded = []
    for symptom in symptoms:
        if symptom in symptom_weights['Symptom'].values:
            weight = symptom_weights.loc[symptom_weights['Symptom'] == symptom, 'weight'].values[0]
            symptoms_encoded.append(weight)
        else:
            symptoms_encoded.append(0)  # Si le symptôme n'est pas reconnu, assigner 0
    # Compléter avec des zéros si moins de 17 symptômes
    while len(symptoms_encoded) < 17:
        symptoms_encoded.append(0)
    return np.array(symptoms_encoded).reshape(1, -1)

# Point de terminaison pour prédire la maladie
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Requête reçue")
        data = request.get_json()
        print("Données reçues :", data)

        symptoms = data.get("symptoms", [])
        if not symptoms:
            return jsonify({"error": "Aucun symptôme fourni"}), 400

        print("Symptômes reçus :", symptoms)

        # Encodage des symptômes
        encoded_symptoms = encode_symptoms(symptoms)
        print("Symptômes encodés :", encoded_symptoms)

        # Prédiction
        probabilities = model.predict_proba(encoded_symptoms)[0]
        prediction = model.predict(encoded_symptoms)[0]
        predicted_probability = probabilities[model.classes_.tolist().index(prediction)]
        print("Maladie prédite :", prediction)
        print("Probabilité associée :", predicted_probability)

        # Récupération de la description et des précautions
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
        print("Erreur :", e)  # Ajoutez ce log pour comprendre l'erreur
        return jsonify({"error": str(e)}), 500

# Lancer l'application
if __name__ == '__main__':
    app.run(debug=True)
