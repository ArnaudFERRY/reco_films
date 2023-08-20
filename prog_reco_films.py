import streamlit as st
from PIL import Image
import prepa_data as data
import model_reco as reco

### Personnalisation de la page
# Page setting
st.set_page_config(layout="wide")

### Affichage de la page
# Row A : logo
a1 = st.columns(1)
a1[0].image(Image.open('logo.jpg'))

# Row B : formulaires + bouton
b1, b2, b3, b4 = st.columns([1, 1, 1, 1])

critere_titre = b1.text_input("Film :")
critere_acteur = b2.text_input("Acteur, actrice, réalisateur :")

# Afficher le selectbox des genres : vide au départ
placeholder = b3.empty()
critere_genre = placeholder.selectbox("Genre :", [])

# Creation du bouton
with b4:
    # Espace vides pour le centrer
    st.write("")
    st.write("")
    valider_button = st.button("Valider")
    st.write("")  


### Partie chargement du fichier et initialisation du modèle mise en cache
@st.cache_data
def load_data():
    # Création d'une zone de texte
    message_placeholder = st.empty()
       
    # Chargement des films
    film_selection = data.df_selection_csv_films()
    
    # Initialisation du modèle
    message_placeholder.write("Veuillez patienter...Préparation des données...")
    myKN, film_selection_dummies, film_selection_Xscaled = data.init_KN(film_selection)
    message_placeholder.empty()

    # Chargement de la liste des genres
    list_of_genres = reco.list_genres(film_selection)
    list_of_genres.insert(0, "")

    return film_selection_dummies, film_selection_Xscaled, myKN, list_of_genres

# Appel de la partie mise en cache pour charger les données
film_selection_dummies, film_selection_Xscaled, myKN, list_genres = load_data()

# Mise à jour du selectbox avec la liste des genres
critere_genre = placeholder.selectbox("Genre :", list_genres)

# Zone du bas : Affichage des résultats
# Execution de la fonction de recommandation si le bouton est cliqué
if critere_titre and valider_button:
    reco = reco.reco(film_selection_dummies, film_selection_Xscaled, myKN, critere_titre, critere_genre, critere_acteur)
    st.write("Films recommandés :")
    
    # Paramétrer la colonne Film comme index
    reco = reco.set_index(reco.columns[0])
    
    # Formater les colonnes numériques pour l'affichage
    numeric_cols = ['Duree', 'Note moyenne', 'Annee']
    formatted_reco = reco.copy()
    for col in numeric_cols:
        formatted_reco[col] = formatted_reco[col].apply(lambda x: f'{x:.0f}' if x == int(x) else f'{x:.2f}')
    st.table(formatted_reco)
    

        
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
