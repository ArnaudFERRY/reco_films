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

RQ : Le fichier films_selection.csv est un fichier epuré d'une sélection de films obtenu à partir des fichiers IMDB.

Les différents fichiers sont importés, nettoyés et filtrés afin d'obtenir une sélection de films
Source IMDB : https://datasets.imdbws.com
