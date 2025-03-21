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
        # Agrégation par vote majoritaire
        majority_class = Counter(predictions).most_common(1)[0][0]

        # Convertir la classe majoritaire en son index
        majority_class_index = list(data.target_names).index(majority_class)

        # Vérifier si la prédiction est correcte
        if majority_class_index == y_test[i]:
            correct += 1

# Calcul de la précision globale
accuracy = correct / len(y_test)
print(f"Précision du modèle agrégé : {accuracy:.2f}")
