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


### Comparaison des prévisions in sample

from statsmodels.tsa.vector_ar.vecm import VECM
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# 1. Séparation en train/test
# -----------------------------
train_df = df_diff.loc[:'2019-12-31']
test_df = df_diff.loc['2020-01-01':'2022-12-01']


# -----------------------------
# 2. Estimation du modèle VECM
# -----------------------------
vecm = VECM(train_df, k_ar_diff=3, coint_rank=4, deterministic="ci")  
vecm_fit_train = vecm.fit()

# -----------------------------
# 3. Prédiction des différences
# -----------------------------
n_forecast = len(test_df)
forecast_diff_train = vecm_fit_train.predict(steps=n_forecast)

# Reconstitution des index temporels des prévisions
forecast_index = test_df.index

# Reconstitution en niveau
forecast_level = pd.DataFrame(forecast_diff_train, columns=train_df.columns, index=forecast_index)
forecast_level = forecast_level.cumsum() + train_df.iloc[-1]

# -----------------------------
# 4. Calcul IC
# -----------------------------
residuals = pd.DataFrame(vecm_fit_train.resid, columns=train_df.columns)
std_error = residuals['sea_level'].std()

z = norm.ppf(0.975)
ic_upper = forecast_level['sea_level'] + z * std_error
ic_lower = forecast_level['sea_level'] - z * std_error

# -----------------------------
# 5. Tracé graphique comparatif
# -----------------------------
plt.figure(figsize=(12, 6))

# Série réelle (test)
plt.plot(test_df.index, test_df['sea_level'], color='blue', label='Historique (réel)')

# Prévision
plt.plot(forecast_level.index, forecast_level['sea_level'], color='red', label='Prévisions (modèle)')

# Intervalle de confiance
plt.fill_between(forecast_level.index, ic_lower, ic_upper, color='red', alpha=0.3, label='IC 95%')

# Titre, axes, légende
plt.title("Prévisions vs Réalité (2020–2022) – Données non vues à l'entraînement")
plt.xlabel("Date")
plt.ylabel("Niveau de la mer (valeurs centrées/réelles)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()