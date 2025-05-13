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
results_johansen = test_johansen(df_base)
'''print(results_johansen)'''
# -> effectué, résultat sauvegardé sous le nom 'Test_Johansen_det_order1' dans les images


## Test des résidus de la régression sur les variables I(1) (Engle-Granger)
results_engle_granger = test_residus_engle_granger(df_base)
'''print(results_engle_granger)'''
# -> effectué, résultat sauvegardé sous le nom 'Test_engle_granger' dans les images