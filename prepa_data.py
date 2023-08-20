# Fonctions pour le téléchargement, préparation des données, initialisation du modèle de recommandation

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Fonction de chargement de la sélection de films à partir d'un fichier Excel
# Fichier issu du téléchargement + sélection + nettoyage
def df_selection_csv_films():
    data_file = pd.read_csv('films_selection.csv')
    film_selection = pd.DataFrame(data_file)
    # Remplace les valeurs NaN (issues du csv) par des valeurs vides dans le DataFrame film_selection
    film_selection.fillna('', inplace=True)
    return film_selection


# Initialise le modèle de machine learning Classifier à partir d'un DataFrame
# Retourne le modèle de machine learning, le DataFrame avec dummies et le Dataframe avec variables standardisées (X_scaled)
def init_KN(film_selection):
    
  # Ajout des variables genres
  dm_film_selection = pd.concat([film_selection, film_selection['genres'].str.get_dummies(sep=',')],axis=1)
  # Ajout des variables actor
  dm_film_selection = pd.concat([dm_film_selection, dm_film_selection['actor'].str.get_dummies(sep=',')],axis=1)
  # Ajout des variables actress
  dm_film_selection = pd.concat([dm_film_selection, dm_film_selection['actress'].str.get_dummies(sep=',')],axis=1)
  # Ajout des variables realisateur
  dm_film_selection = pd.concat([dm_film_selection, dm_film_selection['director'].str.get_dummies(sep=',')],axis=1)

  # Préparation des variables
  X = dm_film_selection.select_dtypes(include = 'number')
  y = dm_film_selection['title']
  y = pd.DataFrame(y)

  # Standardisation des variables numériques
  X_scaled = StandardScaler().fit_transform(X)
  X_scaled = pd.DataFrame(X_scaled)
  X_scaled.columns = X.columns
  
  # Remodeler le vecteur y en tableau 1D
  y = y.values.ravel()

  # Création du modèle
  model_Classifier = KNeighborsClassifier(weights="distance", n_neighbors = 50).fit(X_scaled, y)
  # Réaliser le train_test_split
  X_scaled_train, X_scaled_test, y_train, y_test = train_test_split(X_scaled,y, random_state=42, train_size=0.75)
  
  return model_Classifier, dm_film_selection, X_scaled









