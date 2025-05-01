import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def preparation_donnees(dataframe,colonnes):

    '''
    Fonction qui prépare le jeu de données pour une régression linéaire : Définition de X et Y, normalisation des données et séparation en échantillon de test et d'entrainement
    Renvoie X_train_scaled, X_test_scaled, y_train, y_test
    '''

    # Chargement des données 
    var_explicative = 'sea level corrected ajusted'
    X = dataframe.drop(columns=var_explicative)
    y = dataframe[var_explicative]

    # Séparation entraînement / test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Renvoie de la fonction
    return X_train_scaled, X_test_scaled, y_train, y_test


def regression_ridge(donnees_preparees):

    '''
    Entrainement d'un modèle de regression ridge sur les données mises en entrée.
    Avec choix du meilleur coefficient de pénalisation.
    Renvoie le modèle entrainé sur les données d'entrainement mises en entrée
    '''

    # Récupération des donnée d'entrée
    X_train_scaled, X_test_scaled, y_train, y_test = donnees_preparees

    # Régression Ridge avec validation croisée pour choisir le meilleur alpha
    alphas = np.logspace(-3, 3, 100)
    ridge = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X_train_scaled, y_train)

    return ridge

def recap_ridge(modele_ridge,donnees_preparees):
    
    '''
    Renvoie le coeff de pénalisation optimal, la MSE et le R² du modèle ridge entrainé.
    '''

    # Récupération des donnée d'entrée
    X_train_scaled, X_test_scaled, y_train, y_test = donnees_preparees

    # Prédictions et évaluation
    y_pred = modele_ridge.predict(X_test_scaled)
    print("Meilleur coeff de pénalisation :", modele_ridge.alpha_)
    print("MSE :", mean_squared_error(y_test, y_pred))
    print("R² :", r2_score(y_test, y_pred))

def equation_ridge(modele_ridge,donnees_preparees):

    '''
    Renvoie l'équation de régression correspondante au modèle ridge entrainé.
    '''

    # Récupération des donnée d'entrée
    X_train_scaled, X_test_scaled, y_train, y_test = donnees_preparees

    coefficients = modele_ridge.coef_
    intercept = modele_ridge.intercept_
    variables_explicatives = X_train_scaled.columns  

    equation = "y = {:.3f}".format(intercept)
    for coef, name in zip(coefficients, variables_explicatives):
        sign = '+' if coef >= 0 else '-'
        equation += " {} {:.3f}×{}".format(sign, abs(coef), name)

    return "Équation du modèle Ridge :\n" + equation
