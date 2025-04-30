def desaison(dataframe,colonnes):

    '''
    Fonction qui applique une désaisonnalité sur les colonnes concernées : X_{t} - X_{t-12}
    '''

    for var in colonnes:
        dataframe[f'{var}_seasonal_diff'] = dataframe[var] - dataframe[var].shift(12)

