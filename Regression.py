### Fichier avec les fonctions permettant d'effectuer et de visualiser des régressions

# Imports

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
import pandas as pd

#### Partie n°1 : Préparation des données ####

def preparation_donnees(dataframe,colonnes):

    '''
    Fonction qui prépare le jeu de données pour une régression linéaire : Définition de X et Y, normalisation des données et séparation en échantillon de test et d'entrainement
    Renvoie X_train_scaled, X_test_scaled, y_train, y_test
    '''

    # Chargement des données 
    var_cible = 'sea_level'
    X = dataframe[colonnes]
    y = dataframe[var_cible]

    # Séparation entraînement / test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Renvoie de la fonction
    return X_train_scaled, X_test_scaled, y_train, y_test


#### Partie n°2 : Régressions simples ####

def Regression_simple(dataframe,variable_explicative):

    '''
    Renvoie un graphique récapitulatif de la régression de sea_level sur la variable explicative choisie.
    '''

    # Chargement des données 
    var_cible = 'sea_level'
    X = dataframe[[variable_explicative]]
    y = dataframe[var_cible]

    # Séparation entraînement / test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardisation des données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    # Modèle
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Coefficient de détermination
    r2 = r2_score(y_test, y_pred)

    # P-value du coefficient 
    X_train_sm = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_train_sm).fit()
    p_value = ols_model.pvalues[1]  

    # Valeurs du modèle
    coef = model.coef_[0]
    intercept = model.intercept_

    # Affichage
    plt.figure(figsize=(8, 6))
    plt.scatter(X_all, y_all, color='blue', label='Données réelles')
    plt.plot(X_all, intercept + coef*X_all, color='red', label=f'sea_level = {intercept:.4f} + {coef:.4f}*{variable_explicative}', linewidth=2)
    plt.xlabel(f'{variable_explicative} (standardisée)')
    plt.ylabel('sea_level')
    plt.title(f'Régression linéaire de sea_level sur {variable_explicative}')

    # Ajouter texte avec les métriques
    textstr = '\n'.join((
        f'$R^2$ = {r2:.3f}',
        f'Intercept = {intercept:.3f}',
        f'Coef = {coef:.3f}',
        f'p-value = {p_value:.3e}'))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
               fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

##########################################


#### Partie n°3 : Régressions multiples ####

def Regression_multiple(dataframe):

    '''
    Fonction qui renvoie un résumé de la régression multiple de sea_level sur le reste du Dataset.
    '''

    # Chargement des données 
    var_cible = 'sea_level'
    X = dataframe.drop(columns =['sea_level','year_month'])
    y = dataframe[var_cible]

    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reconvertir en DataFrame pour conserver les noms de colonnes
    X_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # Ajouter constante pour l'intercept
    X_df = sm.add_constant(X_df)

    # Régression
    model = sm.OLS(y, X_df).fit()

    return model.summary()


############################################

