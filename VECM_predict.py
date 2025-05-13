import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank, select_order

# Le tableau différencié commence au 2011-02 et finit au 2022-12
# Le tableau classique commence au 2011-01 et finit au 2022-12

def df_diff():

    '''
    Renvoie le dataframe avec seulement les variables differenciées.
    '''

    df = pd.read_csv('Base_clean.csv')

    df['year_month'] = pd.to_datetime(df['year_month'])
    df.set_index('year_month',inplace=True)

    df_diff =df[['sea_level', 'sea_temperature', 'greenland_mass', 'antarctica_mass']].diff().dropna()

    return df_diff

def modele_VECM(df_diff):

    '''
    Entraine et renvoie un modèle VECM sur les données passées en entrée selon les paramètre trouvés pus haut
    '''

    # Estimer le modèle VECM avec les lags et les relations de cointégration déterminés
    vecm_model = VECM(df_diff, k_ar_diff=3, coint_rank=4, deterministic="ci")

    # Ajuster le modèle
    vecm_fit = vecm_model.fit()

    return vecm_fit

'''# données

df = pd.read_csv('Base_clean.csv')

# étape 1

df['year_month'] = pd.to_datetime(df['year_month'])
df.set_index('year_month',inplace=True)

df_diff =df[['sea_level', 'sea_temperature', 'greenland_mass', 'antarctica_mass']].diff().dropna()

# Estimer le modèle VECM avec les lags et les relations de cointégration déterminés
vecm_model = VECM(df_diff, k_ar_diff=3, coint_rank=4, deterministic="ci")

# Ajuster le modèle
vecm_fit = vecm_model.fit()'''

def print_prev_diff(data_frame,modele):

    '''
    Prévisions out of sample sur la série différenciée, avec graphique.
    '''

    forecast_steps = 12
    forecast = modele.predict(steps=forecast_steps)

    # Mettre les résultats sous forme de DataFrame avec les bons noms de colonnes
    forecast_df = pd.DataFrame(forecast, columns=data_frame.columns)

    # Trouver la dernière date des observations
    last_date = data_frame.index[-1]

    # Créer une série d'index mensuels à partir de la date suivante
    forecast_index = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=forecast_steps, freq='MS')

    # Appliquer cet index à forecast_df
    forecast_df.index = forecast_index

    # Accéder aux prédictions de sea_level
    sea_level_forecast = forecast_df['sea_level']

    # Historique des données 
    historical_values = data_frame['sea_level'] 

    ## Intervalle de confiance (avec variance cumulative)

    # On fixe le niveau de confiance (ici 95%)
    z_score = norm.ppf(0.975)  # ≈ 1.96

    # Récupérer la variance résiduelle du modèle pour chaque variable
    cov_resid = modele.sigma_u  # matrice de variance-covariance des erreurs
    sea_level_var = cov_resid[data_frame.columns.get_loc('sea_level'), data_frame.columns.get_loc('sea_level')]

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
    plt.plot(data_frame.index, historical_values, label='Historique', color='blue')

    # Tracer les prévisions de la variable d'intérêt
    plt.plot(forecast_df.index, sea_level_forecast , label='Prévisions', color='red')

# Ajouter un trait rouge entre le dernier point historique et le premier point prédit
    plt.plot(
        [data_frame.index[-1], forecast_df.index[0]],
        [historical_values.iloc[-1], sea_level_forecast.iloc[0]],
        color='red',
        linestyle='-')

    # Intervalle de confiance
    plt.fill_between(forecast_df.index, lower_bound, upper_bound, color='red', alpha=0.3, label='IC 95%')

    # Personnaliser le graphique
    plt.title('Prévisions avec Intervalle de Confiance à 95% sur série differenciée')
    plt.xlabel('Date')
    plt.ylabel('Valeur de la variable')
    plt.legend()

    # Affichage du graphique
    plt.show()


df_diff = df_diff()

vecm_fit = modele_VECM(df_diff)

print_prev_diff(data_frame=df_diff,modele=vecm_fit)


############## Prévisions sur la série en niveau

###### faire cette fonction

def print_prev_niveau(dataframe, modele):

    '''
    Renvoie un graphique des prédictions out of sample sur la série en niveau
    '''
    return 'Feur'

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





