from Protocole_donnees import *
df = Base_de_donnees()

# visualisation des résidus de la reg multiple

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def test_visuel_heteroscedasticite(df, var_cible='sea_level', colonnes_a_exclure=['year_month']):
    """
    Fonction qui entraîne une régression multiple et affiche un graphique des résidus
    pour détecter visuellement une éventuelle hétéroscédasticité.
    """
    # Préparation des données
    X = df.drop(columns=[var_cible] + colonnes_a_exclure)
    y = df[var_cible]

    # Standardisation des variables explicatives
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ajouter une constante (intercept)
    X_scaled = sm.add_constant(X_scaled)

    # Entraînement du modèle OLS
    model = sm.OLS(y, X_scaled).fit()

    # Affichage du résumé (facultatif)
    print(model.summary())

    # Récupération des résidus et des valeurs prédites
    residuals = model.resid
    fitted_values = model.fittedvalues

    # Graphique des résidus vs. valeurs prédites
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_values, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Valeurs prédites")
    plt.ylabel("Résidus")
    plt.title("Résidus vs Valeurs prédites")
    plt.tight_layout()
    plt.show()

test_visuel_heteroscedasticite(df)


# test BP sur les résidus

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan

def test_breusch_pagan(df, var_cible='sea_level', colonnes_a_exclure=['year_month']):
    """
    Entraîne une régression OLS sur les données et effectue le test de Breusch-Pagan
    pour détecter une hétéroscédasticité.
    """
    # Séparer variables explicatives et cible
    X = df.drop(columns=[var_cible] + colonnes_a_exclure)
    y = df[var_cible]

    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ajout d'une constante
    X_scaled = sm.add_constant(X_scaled)

    # Régression OLS
    model = sm.OLS(y, X_scaled).fit()

    # Test de Breusch-Pagan
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(model.resid, model.model.exog)

    # Affichage des résultats
    print("=== Test de Breusch-Pagan ===")
    print(f"Statistique LM         : {lm_stat:.3f}")
    print(f"p-value (LM)           : {lm_pvalue:.4f}")
    print(f"Statistique F          : {f_stat:.3f}")
    print(f"p-value (F)            : {f_pvalue:.4f}")

    if lm_pvalue < 0.05:
        print("➡️  Il y a des signes d'hétéroscédasticité (p < 0.05).")
    else:
        print("✅ Pas de preuve d'hétéroscédasticité (p ≥ 0.05).")

test_breusch_pagan(df)