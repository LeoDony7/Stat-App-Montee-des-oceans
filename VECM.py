## Modèle VECM 

# import

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank, select_order



# données

df = pd.read_csv('Base_clean.csv')

# étape 1

df['year_month'] = pd.to_datetime(df['year_month'])
df.set_index('year_month',inplace=True)


## Trouver les bons paramètres pour estimer le VECM

#on commence par ne garder que les colonnes I(1) du df puis on les différencie

df_diff =df[['sea_level', 'sea_temperature', 'greenland_mass', 'antarctica_mass']].diff().dropna()

# Choix des paramètres lag et coint rank.

order_res = select_order(df_diff, maxlags=12, deterministic="ci")
print(order_res.summary())

res = select_coint_rank(df_diff, det_order=1, k_ar_diff=3, method="trace", signif=0.05)
print("Nombre de relations de co-intégration :", res.rank)

# on trouve 4 relations de cointégrations avec un lag de 3 -> on va garder ca je pense
# idem avec 5 lags

# donc on va prendre k_ar_diff = 3 et coint_rank = 4

# Estimation du modèle

# Estimer le modèle VECM avec les lags et les relations de cointégration déterminés
vecm_model = VECM(df_diff, k_ar_diff=3, coint_rank=4, deterministic="ci")

# Ajuster le modèle
vecm_fit = vecm_model.fit()

# Résumé du modèle pour voir les résultats
print(vecm_fit.summary())


## Forecasting : premiers pas

'''# Exemple de prévision sur 10 périodes
forecast_steps = 10
forecast = vecm_fit.predict(steps=forecast_steps)

# Affichage des prévisions
print(forecast)
'''

## Prédictions, premiers graphiques

'''import numpy as np


# Exemple de prévisions à partir du modèle VECM
forecast_steps = 10
forecast = vecm_fit.predict(steps=forecast_steps)

# Historique des données (assumant que df contient les données observées)
historical_values = df_diff['sea_level'] 

# Sélection de la variable d'intérêt pour les prévisions (ici la première variable)
variable_of_interest_forecast = forecast[:, 0]  

# Calcul de l'intervalle de confiance à 95% pour cette variable
forecast_std = np.std(variable_of_interest_forecast)  # Erreur standard des prévisions
ci_upper = variable_of_interest_forecast + 1.96 * forecast_std  # Limite supérieure de l'intervalle de confiance
ci_lower = variable_of_interest_forecast - 1.96 * forecast_std  # Limite inférieure de l'intervalle de confiance

# Création de l'index pour les prévisions
forecast_index = pd.date_range(start=df_diff.index[-1], periods=forecast_steps + 1, freq='M')[1:]

# Création du graphique
plt.figure(figsize=(10, 6))

# Tracer les valeurs historiques
plt.plot(df_diff.index, historical_values, label='Historique', color='blue')

# Tracer les prévisions de la variable d'intérêt
plt.plot(forecast_index, variable_of_interest_forecast, label='Prévisions', color='red')

# Tracer l'intervalle de confiance à 95% pour la variable d'intérêt
plt.fill_between(forecast_index, ci_lower, ci_upper, color='red', alpha=0.2, label='Intervalle de confiance 95%')

# Personnaliser le graphique
plt.title('Prévisions avec Intervalle de Confiance à 95%')
plt.xlabel('Date')
plt.ylabel('Valeur de la variable')
plt.legend()

# Affichage du graphique
plt.show()'''



## Prévisions intégrées en niveau

'''# Exemple de prévisions à partir du modèle VECM
forecast_steps = 10
forecast = vecm_fit.predict(steps=forecast_steps)

# Historique des données (assumant que df contient les données observées)
historical_values = df['sea_level']  # Utiliser les valeurs originales de niveau de la mer

# Sélection de la variable d'intérêt pour les prévisions (ici la première variable)
variable_of_interest_forecast = forecast[:, 0]

# Calcul de l'intervalle de confiance à 95% pour cette variable
forecast_std = np.std(variable_of_interest_forecast)  # Erreur standard des prévisions
ci_upper = variable_of_interest_forecast + 1.96 * forecast_std  # Limite supérieure de l'intervalle de confiance
ci_lower = variable_of_interest_forecast - 1.96 * forecast_std  # Limite inférieure de l'intervalle de confiance

# Calcul de la dernière valeur historique (niveau de la mer)
last_historical_value = historical_values.iloc[-1]

# Intégration des prévisions (ajout des différences successives pour obtenir les niveaux réels)
forecast_cumulative = np.cumsum(variable_of_interest_forecast) + last_historical_value

# Réintégration de l'intervalle de confiance
ci_upper_cumulative = np.cumsum(ci_upper) + last_historical_value
ci_lower_cumulative = np.cumsum(ci_lower) + last_historical_value

# Création de l'index pour les prévisions (en supposant que df.index est mensuel)
forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=30), periods=forecast_steps, freq='M')

# Création du graphique
plt.figure(figsize=(10, 6))

# Tracer les valeurs historiques
plt.plot(df.index, historical_values, label='Historique', color='blue')

# Tracer les prévisions de la variable d'intérêt (niveaux de la mer prévus)
plt.plot(forecast_index, forecast_cumulative, label='Prévisions', color='red')

# Tracer l'intervalle de confiance à 95% pour la variable d'intérêt
plt.fill_between(forecast_index, ci_lower_cumulative, ci_upper_cumulative, color='red', alpha=0.2, label='Intervalle de confiance 95%')



# Personnaliser le graphique
plt.title('Prévisions avec Intervalle de Confiance à 95%')
plt.xlabel('Date')
plt.ylabel('Valeur de la variable (niveau de la mer)')
plt.legend()

# Affichage du graphique
plt.show()'''


##