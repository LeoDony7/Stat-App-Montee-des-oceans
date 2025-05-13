import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank, select_order


# données

df = pd.read_csv('Base_clean.csv')

# étape 1

df['year_month'] = pd.to_datetime(df['year_month'])
df.set_index('year_month',inplace=True)

df_diff =df[['sea_level', 'sea_temperature', 'greenland_mass', 'antarctica_mass']].diff().dropna()

# Estimer le modèle VECM avec les lags et les relations de cointégration déterminés
vecm_model = VECM(df_diff, k_ar_diff=3, coint_rank=4, deterministic="ci")

# Ajuster le modèle
vecm_fit = vecm_model.fit()



# Le tableau différencié commence au 2011-02 et finit au 2022-12
# Le tableau classique commence au 2011-01 et finit au 2022-12


# Exemple de prévisions à partir du modèle VECM
forecast_steps = 12
forecast = vecm_fit.predict(steps=forecast_steps)

# Mettre les résultats sous forme de DataFrame avec les bons noms de colonnes
forecast_df = pd.DataFrame(forecast, columns=df_diff.columns)

# Trouver la dernière date de df_diff
last_date = df_diff.index[-1]

# Créer une série d'index mensuels à partir de la date suivante
forecast_index = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=forecast_steps, freq='MS')

# Appliquer cet index à forecast_df
forecast_df.index = forecast_index

# Accéder aux prédictions de sea_level
sea_level_forecast = forecast_df['sea_level']

# Historique des données 
historical_values = df_diff['sea_level'] 

## Intervalle de confiance
# Étape 1 : récupérer la matrice de covariance des résidus
cov_matrix = vecm_fit.sigma_u

# Étape 2 : identifier la position de 'sea_level' dans les colonnes
idx = df_diff.columns.get_loc('sea_level')

# Étape 3 : extraire l'écart-type
std_error = np.sqrt(cov_matrix[idx, idx])

# Étape 4 : calcul de l'intervalle de confiance à 95 %
z_score = 1.96
lower_bound = sea_level_forecast - z_score * std_error
upper_bound = sea_level_forecast + z_score * std_error



# Création du graphique
plt.figure(figsize=(10, 6))

# Tracer les valeurs historiques
plt.plot(df_diff.index, historical_values, label='Historique', color='blue')

# Tracer les prévisions de la variable d'intérêt
plt.plot(forecast_df.index, sea_level_forecast , label='Prévisions', color='red')

# Ajouter un trait rouge entre le dernier point historique et le premier point prédit
plt.plot(
    [df_diff.index[-1], forecast_df.index[0]],
    [historical_values.iloc[-1], sea_level_forecast.iloc[0]],
    color='red',
    linestyle='-')

# Intervalle de confiance
plt.fill_between(forecast_df.index, lower_bound, upper_bound, color='red', alpha=0.3, label='IC 95%')


# Personnaliser le graphique
plt.title('Prévisions avec Intervalle de Confiance à 95%')
plt.xlabel('Date')
plt.ylabel('Valeur de la variable')
plt.legend()

# Affichage du graphique
plt.show()

'''# Calcul de l'intervalle de confiance à 95% pour cette variable
forecast_std = np.std(variable_of_interest_forecast)  # Erreur standard des prévisions
ci_upper = variable_of_interest_forecast + 1.96 * forecast_std  # Limite supérieure de l'intervalle de confiance
ci_lower = variable_of_interest_forecast - 1.96 * forecast_std  # Limite inférieure de l'intervalle de confiance
'''

'''# Tracer l'intervalle de confiance à 95% pour la variable d'intérêt
plt.fill_between(forecast_index, ci_lower, ci_upper, color='red', alpha=0.2, label='Intervalle de confiance 95%')
'''