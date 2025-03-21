from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import requests

# Charger le dataset Iris
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adresses des serveurs Flask
servers = [
    "http://127.0.0.1:5000/predict",  # Code_1 : Régression Logistique
    "http://127.0.0.1:5001/predict",  # Code_2 : Random Forest
    "http://127.0.0.1:5002/predict",  # Code_3 : SVM
    "http://127.0.0.1:5003/predict"   # Code_4 : KNN
]

# Initialiser les poids pour chaque modèle
model_weights = [1.0 for _ in servers]

# Fonction pour interroger les serveurs Flask
def get_predictions(input_data):
    predictions = []
    probabilities_list = []
    for server in servers:
        try:
            response = requests.get(server, params=input_data)
            if response.status_code == 200:
                data = response.json()
                predictions.append(data["prediction"])
                probabilities_list.append(data["probabilities"])
            else:
                print(f"Erreur sur {server}: {response.status_code}")
        except Exception as e:
            print(f"Impossible de se connecter au serveur {server}: {e}")
    return predictions, probabilities_list

# Fonction pour mettre à jour les poids des modèles
def update_weights(predictions, consensus_prediction):
    global model_weights
    for i, pred in enumerate(predictions):
        if pred == consensus_prediction:
            model_weights[i] = min(model_weights[i] + 0.1, 1.0)  # Augmenter le poids si correct
        else:
            model_weights[i] = max(model_weights[i] - 0.1, 0.0)  # Réduire le poids si incorrect

# Tester le modèle agrégé sur les données de test
correct = 0
for i, sample in enumerate(X_test):
    # Préparer les paramètres pour l'API
    params = {
        "sepal_length": sample[0],
        "sepal_width": sample[1],
        "petal_length": sample[2],
        "petal_width": sample[3]
    }
    predictions, probabilities_list = get_predictions(params)

    if predictions and probabilities_list:
        # Agrégation pondérée des probabilités
        avg_probabilities = {cls: 0 for cls in data.target_names}
        for model_idx, probs in enumerate(probabilities_list):
            for cls, prob in probs.items():
                avg_probabilities[cls] += float(prob) * model_weights[model_idx]

        # Normalisation des probabilités
        total_weight = sum(model_weights)
        avg_probabilities = {cls: prob / total_weight for cls, prob in avg_probabilities.items()}

        # Classe avec la probabilité maximale (consensus)
        consensus_prediction = max(avg_probabilities, key=avg_probabilities.get)

        # Convertir la classe consensuelle en son index
        consensus_index = list(data.target_names).index(consensus_prediction)

        # Vérifier si la prédiction consensuelle est correcte
        if consensus_index == y_test[i]:
            correct += 1

        # Mettre à jour les poids en fonction du consensus
        update_weights(predictions, consensus_prediction)

# Calcul de la précision globale
accuracy = correct / len(y_test)
print(f"Précision du modèle agrégé avec pondération : {accuracy:.2f}")
print("Poids finaux des modèles :", model_weights)
