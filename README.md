# 🌡️ Medical Disease Prediction API

## 📌 Description
This API predicts diseases based on symptoms provided by the user. It utilizes a **Random Forest** machine learning model trained on a medical dataset and returns:
- 🏥 A **predicted disease**
- 📊 A **confidence score**
- 📖 A **detailed description**
- ✅ Recommended **precautions**

---
## 🚀 Technologies Used
🔹 Python  
🔹 Flask  
🔹 Pandas  
🔹 NumPy  
🔹 Joblib  
🔹 Matplotlib  
🔹 Seaborn  
🔹 Scikit-learn *(Random Forest Classifier, Model Selection, Metrics)*  

---
## 📊 Model Performance
Our **Random Forest** model was trained and evaluated using cross-validation. Here are the key performance metrics:
- 🎯 **Accuracy**: 99.08%
- 📈 **Precision**: 99.00%
- 🔄 **Recall**: 99.00%
- 📉 **F1-score**: 98.94%

### 🔢 Confusion Matrix
![image](https://github.com/user-attachments/assets/d692e90d-5175-49a1-8260-7f20250570e1)


---
## 🛠️ Installation
### 1️⃣ Clone the repository
```sh
git clone https://github.com/Oussamaroom67/modelApiPredictApp.git
cd modelApiPredictApp
```
### 2️⃣ Install dependencies
```sh
pip install -r requirements.txt
```

---
## 🎯 Usage
### ▶️ Start the API
```sh
python api.py
```
📌 The API will run on:  **`http://127.0.0.1:5000/`**

---
## 🔗 Available Endpoint
### 🔥 `POST /predict`
- **Description**: Predicts a disease based on symptoms.
- **Request Body** (JSON):
  ```json
  {
    "symptoms": ["fatigue", "fever", "cough"]
  }
  ```
- **Response Example** (JSON):
  ```json
  {
    "disease": "Influenza",
    "confidence": 0.87,
    "description": "Influenza is a viral infection that attacks the respiratory system...",
    "precautions": ["Drink plenty of water", "Get rest", "Consult a doctor"]
  }
  ```

---
## 👤 Authors
📝 Project developed by **Oussama Nouhar, Omaima Siaf, and Souhayla Ghanem**.

