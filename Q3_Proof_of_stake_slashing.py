import json
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

# Initialisation des poids et des dépôts pour chaque modèle
model_weights = [1.0 for _ in servers]
initial_deposit = 1000
model_balances = {server: initial_deposit for server in servers}

# Base de données JSON pour suivre les balances
DATABASE_FILE = "model_balances.json"

# Sauvegarder la base de données
def save_database():
    with open(DATABASE_FILE, "w") as f:
        json.dump(model_balances, f, indent=4)

# Charger la base de données
def load_database():
    global model_balances
    try:
        with open(DATABASE_FILE, "r") as f:
            model_balances = json.load(f)
    except FileNotFoundError:
        save_database()

# Charger les données initiales
load_database()

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

# Fonction pour mettre à jour les poids et appliquer les pénalités
def update_weights_and_slash(predictions, consensus_prediction):
    global model_weights, model_balances
    for i, pred in enumerate(predictions):
        server = servers[i]
        if pred == consensus_prediction:
            model_weights[i] = min(model_weights[i] + 0.1, 1.0)  # Augmenter le poids
        else:
            model_weights[i] = max(model_weights[i] - 0.1, 0.0)  # Réduire le poids
            model_balances[server] -= 50  # Appliquer une pénalité de 50 euros
            print(f"Slashing applied to {server}: -50 euros (new balance: {model_balances[server]} euros)")

        # Si le solde tombe à zéro, exclure le modèle
        if model_balances[server] <= 0:
            print(f"{server} has been excluded due to insufficient balance!")
            servers.remove(server)

    save_database()

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

        # Mettre à jour les poids et appliquer les pénalités
        update_weights_and_slash(predictions, consensus_prediction)

# Calcul de la précision globale
accuracy = correct / len(y_test)
print(f"Précision du modèle agrégé avec Proof-of-Stake : {accuracy:.2f}")
print("Poids finaux des modèles :", model_weights)
print("Balances finales des modèles :", model_balances)
