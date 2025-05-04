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

### Test du VIF lorsqu'on crée ice_mass

'''import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Exemple : on souhaite faire la moyenne de var1 et var2
var1 = 'greenland_mass'
var2 = 'antarctica_mass'

# 1. Créer la colonne moyenne
df['ice_mass'] = df[[var1, var2]].mean(axis=1)

# 2. Créer le nouveau DataFrame en supprimant les deux colonnes initiales (et sea_level car variable cible)
df_reduit = df.drop(columns=[var1, var2,'sea_level'])

# 3. Calcul du VIF
X = df_reduit.select_dtypes(include=[float, int]).dropna()  # On garde uniquement les colonnes numériques sans NaN
X = X.reset_index(drop=True)  # S'assurer que l'index est aligné pour statsmodels

# Création du DataFrame pour stocker les VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Affichage des résultats
print(vif_data.sort_values("VIF", ascending=False))'''


### Test du VIF enlevant ice_mass

'''import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Exemple : on souhaite faire la moyenne de var1 et var2
var1 = 'greenland_mass'
var2 = 'antarctica_mass'

# 1. Créer le nouveau DataFrame en supprimant les deux colonnes initiales (et aussi sea_level car variable cible)
df_reduit = df.drop(columns=[var1, var2,'sea_level'])

# 3. Calcul du VIF
X = df_reduit.select_dtypes(include=[float, int]).dropna()  # On garde uniquement les colonnes numériques sans NaN
X = X.reset_index(drop=True)  # S'assurer que l'index est aligné pour statsmodels

# Création du DataFrame pour stocker les VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Affichage des résultats
print(vif_data.sort_values("VIF", ascending=False))'''

#### fonction pour avoir la regression polynomiale avec AIC

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import statsmodels.api as sm
from statsmodels.tools import add_constant

def modele_polynomial_selection(df, cible='sea_level', critere='AIC'):
    """
    Applique une régression polynomiale d'ordre 2 avec sélection de variables par AIC ou BIC.
    
    Paramètres :
    - df : DataFrame contenant les données (y compris la cible et les variables explicatives)
    - cible : nom de la variable cible (par défaut 'sea_level')
    - critere : 'AIC' ou 'BIC' pour la sélection du meilleur modèle
    
    Retour :
    - best_model : modèle statsmodels ajusté
    - selected_features : liste des variables sélectionnées
    """
    
    # 1. Séparation de la cible et des variables explicatives
    y = df[cible]
    X = df.drop(columns=[cible,'greenland_mass','antarctica_mass'])

    # 2. Suppression de la colonne temporelle si elle existe
    if 'year_month' in X.columns:
        X = X.drop(columns=['year_month'])

    # 3. Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # 4. Création des termes quadratiques et d'interaction
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X_scaled_df)
    X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))

    # 5. Réinitialisation des index pour éviter les problèmes dans statsmodels
    X_poly_df = X_poly_df.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # 6. Sélection de variables par forward selection avec AIC/BIC
    def forward_selection(X, y):
        initial_features = []
        remaining_features = list(X.columns)
        selected_features = []
        current_score, best_new_score = np.inf, np.inf
        best_model = None

        while remaining_features:
            scores_with_candidates = []
            for candidate in remaining_features:
                features = selected_features + [candidate]
                X_model = add_constant(X[features])
                model = sm.OLS(y, X_model).fit()
                score = model.aic if critere == 'AIC' else model.bic
                scores_with_candidates.append((score, candidate, model))

            scores_with_candidates.sort()
            best_new_score, best_candidate, best_candidate_model = scores_with_candidates[0]

            if current_score == np.inf or best_new_score < current_score:
                remaining_features.remove(best_candidate)
                selected_features.append(best_candidate)
                current_score = best_new_score
                best_model = best_candidate_model
            else:
                break

        return best_model, selected_features

    # 7. Lancement de la sélection
    best_model, selected_features = forward_selection(X_poly_df, y)

    # 8. Résultats
    print("Variables sélectionnées :", selected_features)
    print("\nRésumé du modèle final :")
    print(best_model.summary())

    return best_model, selected_features

best_model, selected = modele_polynomial_selection(df, cible='sea_level')


