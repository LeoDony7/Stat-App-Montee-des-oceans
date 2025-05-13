### Modèle VECM ###

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


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

