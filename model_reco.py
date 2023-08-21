import pandas as pd

# Retourne la liste des genres du DataFrame
def list_genres(df):
    df_genres = pd.DataFrame(df['genres'])
    df_genres['genres'] = df['genres'].apply(lambda x: x.split(','))
    df_genres = df_genres.explode('genres')
    df_genres.drop_duplicates(subset=['genres'], inplace=True)
    df_genres = df_genres.reset_index()
    df_genres = pd.DataFrame(df_genres['genres'])
    df_genres = df_genres[df_genres['genres'] != 'inconnus']
    df_genres = df_genres.sort_values('genres')
    list_genres = df_genres['genres'].tolist()
    return list_genres

# [Pour DEBUG] Fonction affichage d'un DataFrame
def display_df(df, nb_lignes):
  from tabulate import tabulate
  columns = df.columns.tolist()
  print(tabulate(df.iloc[:nb_lignes, 0:len(columns)], headers=columns, tablefmt='fancy_grid'))


# Fonction pour formater la colonne genre
def format_genre(genres_list):
    if isinstance(genres_list, str):
        # Convertir la chaîne en une liste de genres
        genres_list = genres_list.split(',')
    return ', '.join(genres_list)

### Fonction de recommandation
# Prend en parametre un DataFrame avec dummies un modele ML et des criteres de recherche
# 3 parametres possible : titre (obligatoire pour le moment), genre, acteur
# Retourne un DataFrame avec les films recommandes

# Fonctionnement :
# Calcul de la somme vectorielle de la selection des films correspondant aux criteres
# Calcul des plus proches voisins de cette somme vectorielle
# Tri de la liste selon un nombre de points (ponderation):
  # Selon le film recherche : Genres en commun, acteurs/actrices en commun
  # Selon la personne recherche: acteur ou actrice ou realisateur
  # Selon le genre recherche

def reco(df, X_scaled, model_Classifier, critere_titre, critere_genre, critere_acteur):
  try:
    recherche_titre = df.loc[(df['title'] == critere_titre)]

    # Si plusieurs titres trouves => On prend le plus recent
    if len(recherche_titre) > 1:
      recherche_titre = recherche_titre.sort_values('startYear', ascending = False)
      recherche_titre = recherche_titre.iloc[0:1]

    recherche_titre_scaled = X_scaled.iloc[recherche_titre.index]

  except :
    recherche_titre_scaled = pd.DataFrame()
    recherche_titre = pd.DataFrame()
    print("Pas de critere titre")


  # Traitement du critere Genre
  try:
    recherche_genre = df.loc[(df[critere_genre] == 1)]
    recherche_genre_scaled = X_scaled.iloc[recherche_genre.index]
  except :
    recherche_genre_scaled = pd.DataFrame()

  # # Traitement du critere Acteur
  try:
    recherche_acteur = df.loc[(df[critere_acteur] == 1)]
    recherche_acteur_scaled = X_scaled.iloc[recherche_acteur.index]
  except :
    recherche_acteur_scaled = pd.DataFrame()

  # Calcul du nombre de resultats
  nb_results = len(recherche_titre_scaled) + len(recherche_genre_scaled) + len(recherche_acteur_scaled)
  # print("Nb results :", nb_results)

  # Ajout des ponderations selon les criteres
  recherche_titre_scaled = recherche_titre_scaled * nb_results
  #recherche_acteur_scaled[critere_acteur] = recherche_acteur_scaled[critere_acteur] * 2000

  # Creation d'un DataFrame avec les resultats ponderes
  recherche_scaled = pd.concat([recherche_titre_scaled, recherche_genre_scaled, recherche_acteur_scaled])

  # Recommandation selon la mÃ©thode de la somme vectorielle
  # VÃ©rifie le nombre de fonctionnalites de recherche_scaled
  if recherche_scaled.shape[1] > 0:
    somme_vec = pd.DataFrame(recherche_scaled.sum()).T
  else:
    # Gerer le cas ou recherche_scaled n'a pas de fonctionnalites
    somme_vec = pd.DataFrame()

  if somme_vec.shape[0] > 0:  # Verifie le nombre d'echantillons de somme_vec
    voisins_proches = model_Classifier.kneighbors(somme_vec.to_numpy(), return_distance=False)[0]
  else:
    # Gerer le cas ou somme_vec n'a pas d'echantillon
    voisins_proches = []

  nearest_neighbors_df = df.iloc[voisins_proches]
  # print("Affichage des voisins :")
  # df_nn =   nearest_neighbors_df.iloc[:, [1,2,3,4,5,6,7]]
  # display_df(df_nn, 10)

  if len(recherche_titre) > 0:
      # option_context : pour masquer les messages d'avertissement a l'execution
      with pd.option_context('mode.chained_assignment', None):
            # On classe les plus proches voisins, selon les criteres (points attribues pour chaque critere):

            # Recherche du nombre de genres en commun avec le film
            recherche_titre['genres'] = recherche_titre['genres'].apply(lambda x: tuple(x.split(',')))
            genre_ref = set(genre for genres in recherche_titre['genres'] for genre in genres)
            nearest_neighbors_df['genres'] = nearest_neighbors_df['genres'].apply(lambda x: set(x.split(',')))
            nearest_neighbors_df['nb genres'] = nearest_neighbors_df['genres'].apply(lambda x: len(genre_ref.intersection(x)))

            # Recherche du nombre d'acteurs en commun avec le film
            recherche_titre['actor'] = recherche_titre['actor'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
            
            actor_ref = [actor for actors in recherche_titre['actor'] for actor in actors]
            nearest_neighbors_df['actor'] = nearest_neighbors_df['actor'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
            
            nearest_neighbors_df['nb actors'] = nearest_neighbors_df['actor'].apply(lambda x: len([actor for actor in x if actor in actor_ref]))
            # Appliquer la ponderation
            ponderation = 3
            nearest_neighbors_df['nb actors'] = nearest_neighbors_df['nb actors'].apply(lambda x: x * ponderation)

            # Recherche du nombre d'actrices en commun avec le film
            recherche_titre['actress'] = recherche_titre['actress'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
            
            actor_ref = [actor for actors in recherche_titre['actress'] for actor in actors]
            nearest_neighbors_df['actress'] = nearest_neighbors_df['actress'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)

            nearest_neighbors_df['nb actress'] = nearest_neighbors_df['actress'].apply(lambda x: len([actor for actor in x if actor in actor_ref]))
            # Appliquer la ponderation
            ponderation = 3
            nearest_neighbors_df['nb actress'] = nearest_neighbors_df['nb actress'].apply(lambda x: x * ponderation)

            # Recherche du nombre de realisateurs en commun avec le film
            recherche_titre['director'] = recherche_titre['director'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
            
            actor_ref = [actor for actors in recherche_titre['director'] for actor in actors]
            nearest_neighbors_df['director'] = nearest_neighbors_df['director'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
            
            nearest_neighbors_df['nb real'] = nearest_neighbors_df['director'].apply(lambda x: len([actor for actor in x if actor in actor_ref]))
            # Appliquer la ponderation
            ponderation = 3
            nearest_neighbors_df['nb real'] = nearest_neighbors_df['nb real'].apply(lambda x: x * ponderation)

            # [Si le critere acteur est renseigne] Recherche du nombre d'acteurs en commun avec le critere acteur
            # critere_acteur ici peut etre un acteur, une actrice ou un realisateur
            # Initialisation des points Ã  0
            nearest_neighbors_df['nb crit actors'] = 0
            nearest_neighbors_df['nb crit actress'] = 0
            nearest_neighbors_df['nb crit real'] = 0
            if len(critere_acteur) > 0:
              crit_actor_ref = {critere_acteur}
              # Appliquer la ponderation (crÃ©ation de 3 colonnes pour la ponderation: acteur, actrice et realisateur)
              # Ainsi si un des critÃ¨res correspond avec le film, la pondÃ©ration est appliquÃ©e
              ponderation = 4
              # Recherche si correspondance avec acteur recherche
              nearest_neighbors_df['nb crit actors'] = nearest_neighbors_df['actor'].apply(
                  lambda x: len(crit_actor_ref.intersection(x))*ponderation
              )
              # Recherche si correspondance avec actrice recherchee
              nearest_neighbors_df['nb crit actress'] = nearest_neighbors_df['actress'].apply(
                  lambda x: len(crit_actor_ref.intersection(x))*ponderation
              )
              # Recherche si correspondance avec realisateur recherche
              nearest_neighbors_df['nb crit real'] = nearest_neighbors_df['director'].apply(
                  lambda x: len(crit_actor_ref.intersection(x))*ponderation
              )


            # [Si le critere genre est renseignÃ©] Recherche du nombre de genres en commun avec le critÃ¨re genre
            nearest_neighbors_df['nb crit genre'] = 0
            if len(critere_genre) > 0:
              crit_genre_ref = {critere_genre}
              # Appliquer la pondÃ©ration si correspondance avec genre recherchÃ©
              ponderation = 5
              nearest_neighbors_df['nb crit genre'] = nearest_neighbors_df['genres'].apply(
                  lambda x: len(crit_genre_ref.intersection(x))*ponderation
              )

            # Total des colonnes en commun
            nearest_neighbors_df['som_crit'] = nearest_neighbors_df['nb genres'] + nearest_neighbors_df['nb actors'] + nearest_neighbors_df['nb actress']
            nearest_neighbors_df['som_crit'] += nearest_neighbors_df['nb crit actors'] + nearest_neighbors_df['nb crit actress'] + nearest_neighbors_df['nb crit real'] + nearest_neighbors_df['nb crit genre']

            # print("Affichage de nearest_neighbors_df :")
            # df_nn = nearest_neighbors_df.iloc[:, [1,2,3,4,7,-1,-2,-3,-4,-5,-6,-7,-8,-9]]
            # display_df(df_nn, 10)

            # On ajoute des points pour le film recherchÃ© (1e ligne par defaut avant le tri par som_crit)
            nearest_neighbors_df.iloc[0:1]['som_crit'] += 10

            # On trie par nombre de genres en commun
            nearest_neighbors_df = nearest_neighbors_df.sort_values('som_crit', ascending = False)
            # print("Affichage de nearest_neighbors_df triÃ© par somme de critÃ¨res commun :")
            # df_nn = nearest_neighbors_df.iloc[:, [1,2,3,4,7,-1,-2,-3,-4,-5,-6,-7,-8,-9]]
            # display_df(df_nn, 30)
  else:
      nearest_neighbors_df = nearest_neighbors_df
      
  result = nearest_neighbors_df.iloc[:, [1,2,3,4,5,6,7,8,9]].head(10)
  
  
  ### Mise en forme des colonnes pour le rendu final
  # Renommer les colonnes
  colnames = {'actor': 'Acteur',
              'actress': 'Actrice',
              'director': 'Realisateur',
              'startYear': 'Annee',
              'runtimeMinutes': 'Duree',
              'genres': 'Genres',
              'averageRating': 'Note moyenne',
              'numVotes': 'Nb votes',
              }
  result.rename(columns=colnames, inplace=True)
  
  # Formater la liste des genres (de str a liste)
  result['Genres'] = result['Genres'].apply(format_genre)
  
  return result





