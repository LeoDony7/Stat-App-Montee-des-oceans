from Protocole_donnees import *
from Graphiques_Base import *

BDD = chargerBDD('Base_Statapp.csv')

BDD_desaison = Base_desaison(BDD)
print(BDD_desaison.head(15))



'''tracer_variable(BDD_desaison,'sea_salinity')
tracer_variable(BDD_desaison,'sea_temperature')'''

tracer_df(BDD_desaison)