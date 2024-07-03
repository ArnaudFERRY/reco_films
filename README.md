Projet réalisé en juin / juillet 2023 dans le cadre de la formation Data Analyst à la Wild Code School<br>
Merci également à mon groupe projet : Zeineb, Shaoyang, Angélique et Nell

## Structure du projet
### Données
* films_selection.csv : Contient les données sur les films
* logo.jpg : Logo de la page d'accueil
  
### Fichiers python
* prepa_data.py : Initialise le modèle de machine learning à partir des données
* model_reco.py : Fonction de recommandation : retourne des résultats à partir d'un KNeighborsClassifier + data + critères utilisateur
* prog_reco_films.py : Programme principal (Interface Streamlit, chargement des données, initialisation + déclenchement du modèle)

### Autre
* requirements.txt : Contient les versions utilisées (Obligatoire pour la publication sur Streamlit)
<br>
RQ : Le fichier films_selection.csv est un fichier epuré d'une sélection de films obtenu à partir des fichiers bruts IMDB.<br>

1) Les différents fichiers sont importés, nettoyés et filtrés.<br>
2) La sélection est obtenue par jointure entre certains fichiers selon certains critères :<br>
Sélection d'un TOP 1000 films les mieux notés puis sélection des 150 meilleurs acteurs, actrices et réalisateurs de ce top.<br>
La sélection des films correspond aux films de ces 150 personnes.<br>
(*Source IMDB : https://datasets.imdbws.com*)
