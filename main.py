from Protocole_donnees import *
from Graphiques_Base import *
from Regression import *


##### Test de la pipeline de préparation des données #####

DF = chargerBDD('Base_Statapp.csv')

DF_filtre = Dataframe_filtre_periode(DF,['2011-01-01','2022-12-31'])

DF_propre = Base_nettoye(DF_filtre)

DF_desaison = Base_desaison(DF_propre)

# test. A mettre dans la pipeline si nécéssaire
DF_final = Dataframe_filtre_periode(DF_desaison,['2012-01-01','2022-12-31'])


'''tracer_df(DF_desaison)'''

##########################################################


##### Regression ######

'''doublons = DF[DF.duplicated(subset='year_month', keep=False)]
print(doublons)'''


Regression_simple(DF_final,'sea_temperature')

#######################


##### Tests sur saisonnalité ######

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

###################################