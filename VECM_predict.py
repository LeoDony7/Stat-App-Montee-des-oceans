import pandas as pd
import numpy as np
from scipy.stats import norm
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

####### Prévisions sur la série différenciée

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

## Intervalle de confiance (sans variance cumulée puis avec)
'''# Étape 1 : récupérer la matrice de covariance des résidus
cov_matrix = vecm_fit.sigma_u

# Étape 2 : identifier la position de 'sea_level' dans les colonnes
idx = df_diff.columns.get_loc('sea_level')

# Étape 3 : extraire l'écart-type
std_error = np.sqrt(cov_matrix[idx, idx])

# Étape 4 : calcul de l'intervalle de confiance à 95 %
z_score = 1.96
lower_bound = sea_level_forecast - z_score * std_error
upper_bound = sea_level_forecast + z_score * std_error
'''

# On fixe le niveau de confiance (ici 95%)
z_score = norm.ppf(0.975)  # ≈ 1.96

# Récupérer la variance résiduelle du modèle pour chaque variable
cov_resid = vecm_fit.sigma_u  # matrice de variance-covariance des erreurs
sea_level_var = cov_resid[df_diff.columns.get_loc('sea_level'), df_diff.columns.get_loc('sea_level')]

# Construire des intervalles de confiance croissants
forecast_std_errors = []
for h in range(1, forecast_steps + 1):
    std_h = np.sqrt(h * sea_level_var)
    forecast_std_errors.append(std_h)
forecast_std_errors = np.array(forecast_std_errors)

# Calcul des bornes d'IC
upper_bound = sea_level_forecast + z_score * forecast_std_errors
lower_bound = sea_level_forecast - z_score * forecast_std_errors


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

############## Prévisions sur la série en niveau

# Dernière valeur observée (non différenciée)
last_real_value = df['sea_level'].iloc[-1]

# Prévisions différenciées (issues du modèle)
sea_level_forecast_diff = sea_level_forecast

# Reconstruction de la série non différenciée (prévisions cumulées)
sea_level_forecast_level = last_real_value + np.cumsum(sea_level_forecast_diff)

# Intervalle de confiance dans l’échelle différenciée (déjà calculé)
# forecast_std_errors = [sqrt(h * variance)]

# Cumuler les erreurs standards (attention, on cumule les variances)
forecast_var_cumsum = np.cumsum(forecast_std_errors ** 2)
forecast_std_level = np.sqrt(forecast_var_cumsum)

# Calcul de l'intervalle de confiance sur la série reconstituée
upper_bound_level = sea_level_forecast_level + z_score * forecast_std_level
lower_bound_level = sea_level_forecast_level - z_score * forecast_std_level

# Tracé final
plt.figure(figsize=(10, 6))

# Historique (non différencié)
plt.plot(df.index, df['sea_level'], label='Historique', color='blue')

# Prévisions (reconstituées)
plt.plot(forecast_df.index, sea_level_forecast_level, label='Prévisions', color='red')

# IC sur la série reconstituée
plt.fill_between(forecast_df.index, lower_bound_level, upper_bound_level, color='red', alpha=0.3, label='IC 95%')

# Ajouter un trait rouge entre le dernier point historique et le premier point prédit
plt.plot(
    [df.index[-1], forecast_df.index[0]],
    [df['sea_level'].iloc[-1], sea_level_forecast_level.iloc[0]],
    color='red',
    linestyle='-')


# Titre et légende
plt.title("Prévisions du niveau de la mer avec IC à 95%")
plt.xlabel("Date")
plt.ylabel("sea_level")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
