### Modèle VECM ###

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank, select_order
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

## Etape 1 : tester la stationnarité des séries


def test_stationarity(df, alpha=0.05):

    '''
    Test ADF et KPSS au niveau alpha pour chaque variable de la base de données initiale.
    '''

    results = []

    variables = ['sea_level', 'sea_temperature', 'greenland_mass', 'antarctica_mass', 'chlorophylle', 'CO2', 'sea_salinity']

    for var in variables:
        serie = df[var].dropna()

        # ADF Test
        adf_stat, adf_p, _, _, _, _ = adfuller(serie)
        adf_result = "Stationnaire" if adf_p < alpha else "Non stationnaire"

        # KPSS Test
        try:
            kpss_stat, kpss_p, _, _ = kpss(serie, regression='c', nlags="auto")
            kpss_result = "Non stationnaire" if kpss_p < alpha else "Stationnaire"
        except:
            kpss_stat, kpss_p, kpss_result = None, None, "Erreur"

        results.append({
            'Variable': var,
            'ADF p-value': round(adf_p, 4),
            'ADF conclusion': adf_result,
            'KPSS p-value': round(kpss_p, 4) if kpss_p is not None else 'Erreur',
            'KPSS conclusion': kpss_result
        })

    return pd.DataFrame(results)


## Etape 2 : tester la stationnarité des séries différenciées


def test_stationarity_diff(df, alpha=0.05):

    '''
    Test ADF et KPSS sur les variables différenciées (on ne s'intéresse qu'aux variables non stationnaires).
    '''

    results = []
    
    variables = ['sea_level', 'sea_temperature', 'greenland_mass', 'antarctica_mass']

    for var in variables:
        diff_series = df[var].diff().dropna()

        # ADF
        adf_stat, adf_p, _, _, _, _ = adfuller(diff_series)
        adf_result = "Stationnaire" if adf_p < alpha else "Non stationnaire"

        # KPSS
        try:
            kpss_stat, kpss_p, _, _ = kpss(diff_series, regression='c', nlags="auto")
            kpss_result = "Non stationnaire" if kpss_p < alpha else "Stationnaire"
        except:
            kpss_p, kpss_result = None, "Erreur"

        results.append({
            'Variable': var,
            'ADF p-value': round(adf_p, 4),
            'ADF conclusion': adf_result,
            'KPSS p-value': round(kpss_p, 4) if kpss_p is not None else 'Erreur',
            'KPSS conclusion': kpss_result
        })

    return pd.DataFrame(results)


## Etape 3 : Tester le nombre de relations de cointégration

# Méthode 1 : Test de Johansen

def test_johansen(df):

    '''
    Test du nombre de relation de cointégration via le test de Johansen.
    On fixe det_order = 1 (Constante dans l'équation de co-intégration uniquement) et k_ar_diff = 1 pour avoir des résultats explicites au test.
    L'idéal serait de prendre det_order = 4 (présence d'une tendance pour chaque variable) qui correspond plus à nos données, mais dans ce cas on ne connait pas les valeurs de seuil pour la statistique de test.
    Il faudrait également déterminer le k_ar_diff optimal mais nous ne l'avons pas fait.
    '''

    # Préparation des données
    variables = ['sea_level', 'sea_temperature', 'greenland_mass', 'antarctica_mass']
    df_johansen = df[variables].dropna()

    # test de johansen
    jres = coint_johansen(df_johansen, det_order=1, k_ar_diff=1)

    # Affichage des résultats
    print("Statistique de trace (trace statistic):")
    print(jres.lr1)
    print("\nValeurs critiques (critical values):")
    print(jres.cvt)

    # Interprétation rapide
    for i, stat in enumerate(jres.lr1):
        print(f"\nHypothèse H0 : nombre de relations de co-intégration ≤ {i}")
        print(f"Statistique de trace : {stat:.2f}")
        print(f"Valeurs critiques 90% / 95% / 99% : {jres.cvt[i]}")
        if stat > jres.cvt[i, 1]:  # 95% level
            print("→ Rejet de H0 au seuil de 5% : il existe au moins", i+1, "relation(s) de co-intégration.")
        else:
            print("→ H0 non rejetée au seuil de 5%.")

# Méthode 2 : Test des résidus de la régression linéaire

def test_residus_engle_granger(df, alpha=0.05):

    '''
    Effectue un test de Engle Granger.
    '''

    # Chargement des données 
    var_cible = 'sea_level'
    variables = ['sea_level', 'sea_temperature', 'greenland_mass', 'antarctica_mass']
    X = df[variables]
    y = df[var_cible]

    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reconvertir en DataFrame pour conserver les noms de colonnes
    X_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # Ajouter constante pour l'intercept
    X_df = sm.add_constant(X_df)

    # Régression
    model = sm.OLS(y, X_df).fit()

    # Résidus de la régression
    residuals = model.resid

    # Test de stationnarité des résidus avec ADF
    adf_test = adfuller(residuals)
    print(f"\nTest ADF pour les résidus :")
    print(f"Statistique ADF : {adf_test[0]}")
    print(f"p-value : {adf_test[1]}")

    if adf_test[1] < alpha:
        print(f"Les résidus sont stationnaires (p-value < {alpha}).")
    else:
        print(f"Les résidus ne sont pas stationnaires (p-value > {alpha}).")


## Etape 4 : déterminer les paramètres du modèle VECM

def selection_lag_order(df):

    '''
    Selection du nombre de lag optimal pour le modèle VECM via les critères AIC, BIC, FPE et HQIC
    '''
    
    # Préparation des données
    variables = ['sea_level', 'sea_temperature', 'greenland_mass', 'antarctica_mass']
    df_selection = df[variables].dropna()

    # Choix du nombre de lag
    order_res = select_order(df_selection, maxlags=12, deterministic="ci")
    return order_res.summary()


def selection_rank_coint(df,lag_opti):

    '''
    Selection du nombre de relations de cointégration, étant donné le lag optimal.
    On choisit det_order = 1 pour les mêmes raisons que précédemment.
    '''

    # Préparation des données
    variables = ['sea_level', 'sea_temperature', 'greenland_mass', 'antarctica_mass']
    df_selection = df[variables].dropna()

    res = select_coint_rank(df_selection, det_order=1, k_ar_diff=lag_opti, method="trace", signif=0.05)
    return res.rank


## Etape 5 : Entrainement du modèle VECM

def VECM_entraine(df):

    '''
    Entraine un modèle VECM sur les données mises en entrée.
    Paramètres retenus pour le modèle :
     - nombre de lag :3
     - rang de cointégration : 3

    On fixe deterministic = 'lo' , ce qui signifie qu'on suppose des constantes et des tendances linéaires potentiellement différentes pour chaque série.
    '''

    # Préparation des données
    variables = ['sea_level', 'sea_temperature', 'greenland_mass', 'antarctica_mass']
    df_entrainement = df[variables].dropna()

    # Entrainement
    vecm_model = VECM(df_entrainement,k_ar_diff=3,coint_rank=3,deterministic='lo')
    vecm_fit = vecm_model.fit()

    return vecm_fit


## Etape 6 : Prévisions out of sample 


def plot_vecm_predictions(df, vecm_fit, steps=12):

    '''
    Affiche les valeurs observées de 'sea_level' suivies des prédictions du VECM sur les 12 mois suivants.
    
    - Données observées en bleu
    - Prédictions en rouge
    - Intervalle de confiance à 95% (cumulatif)
    '''

    # 1. Préparation des données
    df = df.copy()
    df['year_month'] = pd.to_datetime(df['year_month'])
    df.set_index('year_month', inplace=True)

    df_diff = df[vecm_fit.names].diff().dropna()
    last_date = df_diff.index[-1]

    # 2. Prédictions différenciées
    forecast_diff = vecm_fit.predict(steps=steps)
    forecast_df = pd.DataFrame(forecast_diff, columns=vecm_fit.names)
    forecast_index = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=steps, freq='MS')
    forecast_df.index = forecast_index

    # 3. Conversion en niveau
    last_real_value = df['sea_level'].iloc[-1]
    forecast_level = forecast_df['sea_level'].cumsum() + last_real_value

    # 4. Intervalle de confiance à 95% (cumulatif)
    residuals = pd.DataFrame(vecm_fit.resid, columns=vecm_fit.names)
    std_error = residuals['sea_level'].std()
    z = norm.ppf(0.975)
    
    cumulative_std = np.sqrt(np.cumsum(np.full(steps, std_error**2)))
    ic_upper = forecast_level + z * cumulative_std
    ic_lower = forecast_level - z * cumulative_std

    # 5. Tracé du graphique
    plt.figure(figsize=(12, 6))
    
    # Observations réelles
    plt.plot(df.index, df['sea_level'], color='blue', label='Historique')
    
    # Prédictions
    plt.plot(forecast_df.index, forecast_level, color='red', label='Prévisions')
    
    # Trait entre dernière observation et première prédiction
    plt.plot([df.index[-1], forecast_df.index[0]],
             [df['sea_level'].iloc[-1], forecast_level.iloc[0]],
             color='red', linestyle='-')

    # Intervalle de confiance
    plt.fill_between(forecast_df.index, ic_lower, ic_upper, color='red', alpha=0.3, label='IC 95%')

    # Mise en forme
    plt.title("Prévisions de 'sea_level' avec VECM (niveau + IC 95%)")
    plt.xlabel("Date")
    plt.ylabel("Niveau de la mer")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


## Etape 7 : Prévisions in sample + Evaluation

def evaluate_vecm_with_model(df, vecm_fit, variables, forecast_steps=12):

    '''
    Évalue un modèle VECM déjà fitté via prévision hors échantillon :
    - Prédit forecast_steps mois après 2019-12-01
    - Compare aux vraies valeurs
    - Affiche RMSE, MAE + graphique avec IC 95%

    Paramètres :
    df : DataFrame contenant 'year_month' + colonnes utilisées dans le modèle
    vecm_fit : modèle VECM déjà ajusté
    variables : liste des noms de colonnes utilisées dans le modèle
    forecast_steps : nombre de périodes à prédire (par défaut 12)
    '''

    df = df.copy()
    df['year_month'] = pd.to_datetime(df['year_month'])
    df.set_index('year_month', inplace=True)

    # Définir les bornes temporelles
    test_start = pd.Timestamp("2020-01-01")
    df_train = df[df.index < test_start]
    df_test = df[df.index >= test_start][:forecast_steps]

    # Dernières valeurs avant prévision
    df_train_model = df_train[variables]
    last_values = df_train_model.iloc[-1]

    # Prédictions différenciées
    forecast_diff = vecm_fit.predict(steps=forecast_steps)
    forecast_df = pd.DataFrame(forecast_diff, columns=variables)
    forecast_df.index = pd.date_range(start=last_values.name + pd.offsets.MonthBegin(1),
                                      periods=forecast_steps, freq='MS')

    # Reconstitution en niveau
    forecast_level = forecast_df.cumsum() + last_values

    # Vraies valeurs à comparer
    actual = df_test['sea_level']
    predicted = forecast_level['sea_level']

    # IC 95% : cumul des erreurs
    residuals = pd.DataFrame(vecm_fit.resid, columns=variables)
    std_error = residuals['sea_level'].std()
    z = norm.ppf(0.975)
    cumulative_std = np.sqrt(np.cumsum(np.full(forecast_steps, std_error**2)))
    ic_upper = predicted + z * cumulative_std
    ic_lower = predicted - z * cumulative_std

    # Calcul des scores
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    # Affichage
    print(f"\nÉvaluation des prévisions VECM (sur {forecast_steps} mois):")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAE  : {mae:.3f}")

    # Graphique
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['sea_level'], color='blue', label='Historique')
    plt.plot(predicted.index, predicted, color='red', label='Prévisions')
    plt.fill_between(predicted.index, ic_lower, ic_upper, color='red', alpha=0.3, label='IC 95%')
    plt.title("Prévision in sample de 'sea_level' (VECM)")
    plt.xlabel("Date")
    plt.ylabel("Niveau de la mer")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return predicted, actual, rmse, mae

