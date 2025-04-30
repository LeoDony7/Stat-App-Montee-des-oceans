import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def preparation_donnees(dataframe,colonnes,split=True):

    '''
    Fonction qui prépare le jeu de données pour une régression linéaire : Définition de X et Y, normalisation des données et séparation en échantillon de test et d'entrainement si besoin

    '''

    # Chargement des données 
    var_explicative = 'sea level corrected ajusted'
    X = dataframe.drop(columns=var_explicative)
    y = dataframe[var_explicative]

    ###### Continuez à partir de là

# Séparation entraînement / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Régression Ridge avec validation croisée pour choisir le meilleur alpha
alphas = np.logspace(-3, 3, 100)
ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(X_train_scaled, y_train)

# Prédictions et évaluation
y_pred = ridge.predict(X_test_scaled)
print("Meilleur alpha :", ridge.alpha_)
print("MSE :", mean_squared_error(y_test, y_pred))
print("R² :", r2_score(y_test, y_pred))



