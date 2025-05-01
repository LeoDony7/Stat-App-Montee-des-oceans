## Protocole à appliquer à notre base de données pour pouvoir faire des régressions et bosser dessus correctement

# Imports
import os
import pandas as pd


### Première étape : récupération de la base de données initiale

def chargerBDD(nom_fichier):

    '''
    Fonction qui récupère le fichier de la base de données et renvoie un Dataframe pandas.
    '''

    chemin_script = os.path.dirname(__file__)
    chemin_csv = os.path.join(chemin_script,nom_fichier)
    ## Version alternative si on déplace le fichier
    ## chemin_csv = os.path.join(chemin_script, '..', 'data', nom_fichier)  # fichier dans le dossier "data"
    chemin_csv = os.path.abspath(chemin_csv)

    return pd.read_csv(chemin_csv)


## Deuxième étape : renommer les colonnes

def formatage_date(dataframe):

    '''
    Modifie le DataFrame en entrée pour que la date soit mise au format YYYY-MM-DD.
    '''
    
    # Modification du format de la date
    dataframe['year_month'] = pd.to_datetime(dataframe['year_month'], format='%Y-%m-%d')


def rename_colonnes(dataframe):

    '''
    Renomme les colonnes du Dataframe selon le dictionnaire spécifié ci dessous, dans un but de lisibilité.
    Modifie le Dataframe passé en entrée.
    ''' 

    # Nouveaux noms
    nouveaux_noms = {'sea level corrected adjusted' : 'sea_level',
                     'chlorophylle' : 'chlorophylle_need_desaison',
                     'sst_anomaly_filtered':'sea_temperature',
                     'CO2':'CO2_need_desaison',
                     'salinité': 'sea_salinity_need_desaison'}

    # Changement des noms
    dataframe.rename(columns=nouveaux_noms, inplace=True)


## Troisème étape : Gérer les données manquantes

def interpolation(dataframe):

    '''
    Fonction qui interpole les données manquantes sur notre jeu de données.
    Modifie le dataframe passé en entrée.
    '''

    dataframe['CO2_need_desaison'] = dataframe['CO2_need_desaison'].interpolate(method = 'linear')
    dataframe['greenland_mass'] = dataframe['greenland_mass'].interpolate(method = 'linear')
    dataframe['antarctica_mass'] = dataframe['antarctica_mass'].interpolate(method = 'linear')
    ## Voir comment interpoler avec 'pchip' pour les 2 derniers


## Quatrième étape : Désaisonnaliser les colonnes 

def desaison(dataframe):

    '''
    Fonction qui applique une désaisonnalisation sur toutes les colonnes du DataFrame contenant le suffixe '_need_desaison'.
    Le résultat est stocké dans une nouvelle colonne sans ce suffixe.
    On applique la désaisonnalisation suivante : X_{t} - X_{t-12}
    '''

    for col in dataframe.columns:
        if col.endswith('_need_desaison'):
            nom_base = col.replace('_need_desaison', '')
            dataframe[nom_base] = dataframe[col] - dataframe[col].shift(12)

## Récupération de la base désaisonnalisée

def Base_desaison(dataframe):

    '''
    Création d'un Dataframe avec seulement les colonnes désaisonnalisées.
    En appliquant les fonctions précédentes.
    '''

    # Application des transformations précédentes
    formatage_date(dataframe)

    rename_colonnes(dataframe)

    interpolation(dataframe)

    desaison(dataframe)

    # Suppression des colonnes non-désaisonnalisées
    colonnes_a_supprimer = [col for col in dataframe.columns if col.endswith('_need_desaison')]

    return dataframe.drop(columns=colonnes_a_supprimer)


## Cinquième étape : Filtrer sur la période commune

def Dataframe_filtre_periode(dataframe):

    '''
    Fonction qui renvoie un Dataframe restreint à la période sur laquelle on possède toutes les données.
    '''

    # Selection de la période : du 1 Janvier 2011 au 31 décembre 2022
    filtre = (dataframe['year_month'] >= '2011-01-01') & (dataframe['year_month'] <= '2022-12-31')

    return dataframe[filtre].sort_values('year_month')




