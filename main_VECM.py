from VECM_clean import *

df_base = pd.read_csv('Base_clean.csv')

## Test de stationnarité des séries 
'''stationarity_results = test_stationarity(df_base)'''
'''print(stationarity_results)'''
# -> effectué, résultat sauvegardé sous le nom 'Résultats_test_stationnarité' dans les images


## Test de stationnarité des séries désaisonnalisées
'''results_diff = test_stationarity_diff(df_base)'''
'''print(results_diff)'''
# -> effectué, résultat sauvegardé sous le nom 'Résultats_test_stationnarité_diff' dans les images


## Test de Johansen du nombre de relations de cointégration
'''test_johansen(df_base)'''
# -> effectué, résultat sauvegardé sous le nom 'Test_Johansen_det_order1' dans les images


## Test des résidus de la régression sur les variables I(1) (Engle-Granger)
'''test_residus_engle_granger(df_base)'''
# -> effectué, résultat sauvegardé sous le nom 'Test_engle_granger' dans les images


## Choix du nombre optimal de lag
'''nombre_lag = selection_lag_order(df_base)'''
'''print(nombre_lag)'''
# -> effectué, résultat sauvegardé sous le nom 'Choix_lag_optimal' dans les images


## Choix du rang de cointégration optimal
'''rank_opti_1 = selection_rank_coint(df_base,1)
rank_opti_2 = selection_rank_coint(df_base,2)
rank_opti_3 = selection_rank_coint(df_base,3)
rank_opti_4 = selection_rank_coint(df_base,4)
rank_opti_5 = selection_rank_coint(df_base,5)
rank_opti_6 = selection_rank_coint(df_base,6)
rank_opti_7 = selection_rank_coint(df_base,7)
rank_opti_8 = selection_rank_coint(df_base,8)
print('Rang de cointégration selon le lag choisi :')
print(f"Le rang de cointégration pour un lag de 1 est : {rank_opti_1}")
print(f"Le rang de cointégration pour un lag de 2 est : {rank_opti_2}")
print(f"Le rang de cointégration pour un lag de 3 est : {rank_opti_3}")
print(f"Le rang de cointégration pour un lag de 4 (lag optimal) est : {rank_opti_4}")
print(f"Le rang de cointégration pour un lag de 5 est : {rank_opti_5}")
print(f"Le rang de cointégration pour un lag de 6 (lag optimal) est : {rank_opti_6}")
print(f"Le rang de cointégration pour un lag de 7 est : {rank_opti_7}")
print(f"Le rang de cointégration pour un lag de 8 est : {rank_opti_8}")'''

# -> effectué, résultat sauvegardé sous le nom de 'rang_coint_optimal' dans les images

## Entrainement du modèle
modele_entraine =VECM_entraine(df_base)
'''print(modele_entraine.summary())'''
# /!\ Récuperer le summary pour faire des commentaires


## Prévisions out of sample (en niveau)
'''plot_vecm_predictions(df_base,modele_entraine)'''
# -> effectué, résultat sauvegardé sous le nom 'VECM_bien_specifie_prevision' sur le bureau

## Prévisions in sample + Evaluation
'''df_base['year_month'] = pd.to_datetime(df_base['year_month'])
modele_entraine_IS = VECM_entraine(df_base[df_base['year_month'] < pd.Timestamp("2020-01-01")])
pred, actual, rmse, mae = evaluate_vecm_with_model(df_base, modele_entraine_IS, variables=['sea_level', 'sea_temperature', 'greenland_mass', 'antarctica_mass'])

'''
evaluation_Ljung_Box(df_base,modele_entraine)