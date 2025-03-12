import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


dataset_salinite = xr.open_mfdataset(".\my_data_folder\medsea_annual_data\salinite\*.nc", combine="by_coords")
print(dataset_salinite)

'''
# Sélectionner la première profondeur (ex: surface)
salinity_surface = dataset_salinite['so'].isel(time=0, depth=0)  # Remplace 'ta_variable' par le vrai nom

# Vérifier les dimensions après sélection
print(salinity_surface.dims)  # Doit afficher ('lat', 'lon')

# Récupérer les coordonnées
lon = salinity_surface.coords['lon']
lat = salinity_surface.coords['lat']

# Création de la figure
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))

# Affichage des données sous forme de carte
mesh = ax.pcolormesh(lon, lat, salinity_surface, cmap='viridis', shading='auto', transform=ccrs.PlateCarree())

# Ajouter une barre de couleurs
plt.colorbar(mesh, ax=ax, label='Salinité')

# Ajouter les côtes
ax.coastlines()

# Ajouter un titre
plt.title("Carte de la salinité à la surface (t=0, profondeur=0)")

# Afficher
plt.show()
'''
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.widgets import Slider

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.widgets import Slider

# Charger les données
salinity_data = dataset_salinite['so']  # Remplace 'ta_variable' par le bon nom

# Récupérer les coordonnées
lon = salinity_data.coords['lon']
lat = salinity_data.coords['lat']
time_steps = salinity_data.coords['time'].values  # Tous les instants disponibles
depth_levels = salinity_data.coords['depth'].values  # Toutes les profondeurs disponibles

# Création de la figure
fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(bottom=0.3)  # Espace pour les curseurs

# Affichage initial
time_idx = 0
depth_idx = 0
mesh = ax.pcolormesh(lon, lat, salinity_data.isel(time=time_idx, depth=depth_idx), 
                      cmap='viridis', shading='auto', transform=ccrs.PlateCarree())
ax.coastlines()
plt.colorbar(mesh, ax=ax, label='Salinité')

# Titre mis à jour avec le temps et la profondeur
title = ax.set_title(f"Salinité à {time_steps[time_idx]} | Profondeur: {depth_levels[depth_idx]} m")

# Curseur pour le temps
ax_slider_time = plt.axes([0.2, 0.15, 0.6, 0.03])
slider_time = Slider(ax_slider_time, 'Temps', 0, len(time_steps)-1, valinit=time_idx, valstep=1)

# Curseur pour la profondeur
ax_slider_depth = plt.axes([0.2, 0.05, 0.6, 0.03])
slider_depth = Slider(ax_slider_depth, 'Profondeur', 0, len(depth_levels)-1, valinit=depth_idx, valstep=1)

# Fonction de mise à jour
def update(val):
    time_idx = int(slider_time.val)
    depth_idx = int(slider_depth.val)
    mesh.set_array(salinity_data.isel(time=time_idx, depth=depth_idx).values.flatten())
    title.set_text(f"Salinité à {time_steps[time_idx]} | Profondeur: {depth_levels[depth_idx]} m")
    fig.canvas.draw_idle()

# Lier les curseurs aux mises à jour
slider_time.on_changed(update)
slider_depth.on_changed(update)

# Affichage interactif
plt.show()



