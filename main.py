from Protocole_donnees import *
from Graphiques_Base import *
from Regression import *


##### Test de la pipeline de préparation des données #####

df = Base_de_donnees()

'''tracer_df(df)
'''
##########################################################

##### Regression simple ######

'''
DF = chargerBDD('Base_Statapp.csv')

DF_propre = Base_nettoye(DF)

DF_desaison = Base_desaison(DF_propre)

DF_final = Base_filtre(DF_desaison,['sea_level','chlorophylle'])
Regression_simple(DF_final,'chlorophylle')
'''

##############################

##### Régression multiple #####

'''
DF = chargerBDD('Base_Statapp.csv')

DF_propre = Base_nettoye(DF)

DF_desaison = Base_desaison(DF_propre)

DF_final = Base_filtre(DF_desaison)


import matplotlib.pyplot as plt

def save_model_summary_as_image(summary, filename="regression_summary.png"):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')  # Pas d'axes visibles
    ax.text(0, 1, str(summary), fontsize=10, family='monospace', verticalalignment='top')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

summary = Regression_multiple(DF_final)
save_model_summary_as_image(summary)
'''


###############################

# Regression polynomiale
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
import statsmodels.api as sm

def regression_polynomiale_deg2(dataframe, cible, exclure=[]):
    """
    Régression multiple avec interactions polynomiales de degré 2.

    Paramètres :
    - dataframe : DataFrame d'entrée
    - cible : nom de la variable cible
    - exclure : colonnes à exclure des prédicteurs (ex. colonne date)

    Retour :
    - Résumé du modèle OLS avec noms de variables explicites
    """
    # Séparer X et y
    X = dataframe.drop(columns=[cible] + exclure)
    y = dataframe[cible]

    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Création des interactions polynomiales
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_array = poly.fit_transform(X_scaled)
    feature_names = poly.get_feature_names_out(input_features=X.columns)

    # Alignement des index avec y
    X_poly = pd.DataFrame(X_poly_array, columns=feature_names, index=y.index)

    # Ajouter la constante pour l'intercept
    X_poly = sm.add_constant(X_poly)

    # Régression OLS
    model = sm.OLS(y, X_poly).fit()

    return model.summary()


summary = regression_polynomiale_deg2(df, cible='sea_level', exclure=['year_month'])
print(summary)'''


## Regression polynomiale + Lasso

'''
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import pandas as pd

def regression_lasso_poly(dataframe, cible='sea_level', exclure=['year_month'], degree=2):
    # Séparer X et y
    X = dataframe.drop(columns=[cible] + exclure)
    y = dataframe[cible]

    # Création du pipeline : PolynomialFeatures + Standardisation + LassoCV
    pipeline = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LassoCV(cv=5, random_state=42)
    )
    
    # Entraînement du modèle
    pipeline.fit(X, y)

    # Prédiction + score
    y_pred = pipeline.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"R² du modèle Lasso (interactions degré {degree}) : {r2:.4f}")

    # Récupérer les noms des variables
    poly_features = pipeline.named_steps['polynomialfeatures'].get_feature_names_out(X.columns)
    coefs = pipeline.named_steps['lassocv'].coef_
    
    # Filtrer les variables sélectionnées
    selected = pd.Series(coefs, index=poly_features)
    selected_nonzero = selected[selected != 0].sort_values(key=abs, ascending=False)
    print("\nVariables sélectionnées (coefs non nuls) :")
    print(selected_nonzero)

    return selected_nonzero

regression_lasso_poly(df)
'''