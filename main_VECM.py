from VECM_clean import *

df_base = pd.read_csv('Base_clean.csv')

## Test de stationnarité des séries 
stationarity_results = test_stationarity(df_base)
'''print(stationarity_results)'''
# -> effectué, résultat sauvegardé sous le nom 'Résultats_test_stationnarité' dans les images


## Test de stationnarité des séries désaisonnalisées
results_diff = test_stationarity_diff(df_base)
'''print(results_diff)'''
# -> effectué, résultat sauvegardé sous le nom 'Résultats_test_stationnarité_diff' dans les images


## Test de Johansen du nombre de relations de cointégration
'''test_johansen(df_base)'''
# -> effectué, résultat sauvegardé sous le nom 'Test_Johansen_det_order1' dans les images


## Test des résidus de la régression sur les variables I(1) (Engle-Granger)
'''test_residus_engle_granger(df_base)'''
# -> effectué, résultat sauvegardé sous le nom 'Test_engle_granger' dans les images


## Choix du nombre optimal de lag
nombre_lag = selection_lag_order(df_base)
'''print(nombre_lag)'''
# -> effectué, résultat sauvegardé sous le nom 'Choix_lag_optimal' dans les images


## Choix du rang de cointégration optimal
rank_opti_4 = selection_rank_coint(df_base,4)
rank_opti_6 = selection_rank_coint(df_base,6)
rank_opti_3 = selection_rank_coint(df_base,3)
'''print(f"Le rang de cointégration pour un lag de 4 est : {rank_opti_4}")'''
'''print(f"Le rang de cointégration pour un lag de 6 est : {rank_opti_6}")'''
'''print(f"Le rang de cointégration pour un lag de 3 (compromis) est : {rank_opti_3}")'''
# -> effectué, résultat sauvegardé sous le nom de 'rang_coint_optimal' dans les images

## Entrainement du modèle
modele_entraine =VECM_entraine(df_base)
'''print(modele_entraine.summary())'''
# /!\ Récuperer le summary pour faire des commentaires


## Prévisions out of sample (en niveau)

'''plot_vecm_predictions(df_base,modele_entraine)'''