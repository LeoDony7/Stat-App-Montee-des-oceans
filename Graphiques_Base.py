### Fonctions pour afficher les données.


# Imports
import matplotlib.pyplot as plt


### Tracé d'une seule variable

def tracer_variable(dataframe, colonne):

    '''
    Affiche un graphique de l'évolution temporelle d'une colonne donnée.
    '''

    plt.figure(figsize=(10, 5))
    plt.plot(dataframe['year_month'], dataframe[colonne], label=colonne, linewidth=1.5)
    plt.title(colonne, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(colonne, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()


## Tracé de toutes les variables

def tracer_df(dataframe):

    '''
    Fonction qui trace l'évolution temporelle de toutes les colonnes en mettant les graphiques cote à cote.
    '''

    # Liste des colonnes à tracer
    columns_to_plot = list(dataframe.columns[1:])
    n = len(columns_to_plot)

    # Définir la taille de la figure
    plt.figure(figsize=(14, 10))

    # Tracer chaque colonne
    for i, column in enumerate(columns_to_plot, 1):
        plt.subplot(n // 2 + n % 2, 2, i)
        plt.plot(dataframe['year_month'], dataframe[column], label=column, linewidth=1.5)
        plt.title(column, fontsize=12)
        '''plt.xlabel('Date', fontsize=10)'''
        plt.ylabel(column, fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(fontsize=8,rotation=45)
        plt.tight_layout()

    plt.show()


