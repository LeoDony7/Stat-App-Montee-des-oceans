from Protocole_donnees import *
from Graphiques_Base import *

BDD = chargerBDD('Base_Statapp.csv')

'''BDD_desaison = Base_desaison(BDD)

BDD_filtre = Dataframe_filtre_periode(BDD_desaison)


tracer_variable(BDD_desaison,'sea_temperature')
tracer_df(BDD_desaison)'''

### test sur la désaison

from statsmodels.tsa.seasonal import seasonal_decompose


formatage_date(BDD)

rename_colonnes(BDD)

interpolation(BDD)

BDD = Dataframe_filtre_periode(BDD)


################################# Tests sur saisonnalité

# Désaisonnalisation
decomposition = seasonal_decompose(BDD['CO2_need_desaison'], model='additive', period=12)
BDD['CO2_deseasonalized'] = BDD['CO2_need_desaison'] - decomposition.seasonal

decomp = seasonal_decompose(BDD['CO2_deseasonalized'], model='additive', period=12)
# Visualiser pour vérifier
decomp.plot()
plt.suptitle("Décomposition de la série CO2")
plt.tight_layout()
plt.show()



# Code pour visualiser le truc de saisonnalité et tendance pour toutes les var en même temps.
'''variables = list(BDD.columns[1:])
period = 12

n_vars = len(variables)
fig, axes = plt.subplots(n_vars, 3, figsize=(15, 4 * n_vars), sharex=True)

for i, var in enumerate(variables):
    decomposition = seasonal_decompose(BDD[var], model='additive', period=period)
    BDD[f'{var}_deseasonalized'] = BDD[var] - decomposition.seasonal

    # Affichage
    axes[i, 0].plot(decomposition.trend)
    axes[i, 0].set_title(f"{var} - Tendance")

    axes[i, 1].plot(decomposition.seasonal)
    axes[i, 1].set_title(f"{var} - Saison")

    axes[i, 2].plot(decomposition.resid)
    axes[i, 2].set_title(f"{var} - Résidus")

plt.tight_layout()
plt.suptitle("Décomposition saisonnière de chaque variable", fontsize=16, y=1.02)
plt.show()
'''