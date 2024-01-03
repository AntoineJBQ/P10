import pandas
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def predict(file_path):
    model = joblib.load('modele_svc.sav')  # Charger le modèle
    new_data = pandas.read_csv(file_path)  # Charger les nouvelles données

    # Prétraiter les nouvelles données
    X_new = preprocess(new_data)

    # Ajout de la liste de noms de caractéristiques
    feature_names = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']

    # Renommer les colonnes
    X_new.columns = feature_names

    # Normaliser les nouvelles données
    scaler = StandardScaler()
    scaler.fit(X_new)
    X_new = pandas.DataFrame(scaler.transform(X_new), columns=X_new.columns)
    # faire des prédictions
    y_pred = model.predict(X_new)  

    return y_pred


def preprocess(data):
    # Supprimer la colonne 'id'
    data = data.drop('id', axis=1)
    
    # Vérifier si toutes les données restantes sont de type float
    if not all(data.dtypes == np.float64):
        raise ValueError("Toutes les données doivent être de type float.")
    
    return data

# input("Veuillez entrer le chemin d'accès à votre fichier : ")
if __name__ == "__main__":
    file_path = input("Veuillez entrer le chemin d'accès à votre fichier : ")
    predictions = predict(file_path)
    print("Prédictions :")
    print(predictions)