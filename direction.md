# Fil directeur de notre travail

## 1. Régression simple sur chaque co-variable

On a fait une régression sur chaque facteur pour vérifier empiriquement leur intérêt

-> on voit que certains facteurs semblent plus pertinents que d'autres

## 2. Régression multiple avec toutes les co-variables

-> très mauvais R² et aussi des p-valeurs très élévées pour la plupart des variables

On a donc envie de se lancer dans une modélisation moins linéaire : la régression polynomiale

## 3. Vérification de la corrélation entre nos variables

On fait la matrice de corrélation de nos variables et on voit qu'il y a une forte corrélation entre les fontes de glace et la température (logique).
Dans un premier temps on peut regrouper greenland_mass et antarctica_mass en une moyenne ice_mass car corrélation presque parfaite entre les 2, mais on aura toujours une très forte corrélation entre ice_mass et la température.

Il faut gérer ce problème, il y a 2 approches possibles :
-> soit on supprime une des 2 variables (plutot la fonte des glaces d'après ce que la littérature nous a appris) et ensuite on peut faire des régressions polynomiales sans problème.


