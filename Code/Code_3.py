# Imports

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
import numpy as np
from sklearn.svm import SVC

# Charger et préparer le dataset
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle SVM
model = SVC(probability=True, kernel='rbf')
model.fit(X_train, y_train)

# Créer l'API Flask
app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
    try:
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))

        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        return jsonify({
            "prediction": data.target_names[prediction],
            "probabilities": {
                data.target_names[i]: probabilities[i] for i in range(len(data.target_names))
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)})


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Charger le dataset Iris (déjà utilisé pour l'entraînement)
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Calculer la précision et afficher les métriques
def evaluate_model(model):
    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Précision
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision du modèle : {accuracy:.2f}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatrice de confusion :")
    print(cm)

    # Rapport de classification
    report = classification_report(y_test, y_pred, target_names=data.target_names)
    print("\nRapport de classification :")
    print(report)


# Entraîner le modèle
model = SVC(probability=True, kernel='rbf')
model.fit(X_train, y_train)

# Évaluer le modèle
evaluate_model(model)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
