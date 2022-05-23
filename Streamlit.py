import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

import folium
from folium.plugins import HeatMap
from pathlib import Path


st.title("Musique générale 1950 à 2020")

st.header('1°/ Mes données')

@st.cache
def load_data():
   data = 'tcc_ceds_music.csv'
   data = pd.read_csv(data)
   return data

df = load_data()
st.write(df)

st.header('2°/ Exploration des données')

st.write(df.describe())
st.write("Notre dataset contient donc 31 colonnes et 28372 lignes")
st.write('artist_name :',df['artist_name'].unique())
st.write('release_date :',df['release_date'].unique())
st.write('genre :',df['genre'].unique())

st.write('music.duplicated() ?', df.duplicated().any())
duplicated = df.duplicated()
st.write(df[duplicated])

st.write('il y a pas de duplicated')


st.header('détails sur lartistes')
artists = df["artist_name"].value_counts()[:20].sort_values(ascending = False)
st.write(artists)
st.bar_chart(data = artists , use_container_width=True)

st.header('détails sur topics')
topics = df["topic"].value_counts()[:20].sort_values(ascending = True)
st.write(topics)
st.bar_chart(data = topics , use_container_width=True)

fig, ax = plt.subplots()
sns.set(rc = {'figure.figsize':(20,8)})
sns.barplot(x='genre',y='age',data=df)
st.write(fig)


numeric = df.drop(["artist_name", "track_name", "genre", "lyrics", "topic"], axis = 1)
numeric

plt.clf()
fig, ax = plt.subplots(figsize= (30 , 20))
sns.regplot(x = df['romantic'] , y = df['violence'])
st.write(fig)



plt.clf()
fig, ax = plt.subplots(figsize=(10, 8))
corr = numeric.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
st.write(fig)












data_load_state = st.text('Chargement...')

data = pd.read_csv("top10s.csv", encoding = "ISO-8859-1")
st.write(data.head(20))

data_load_state.text('Chargement...réussi!')


st.write('Ici, nous allons étudier les données de la table "top10s.csv", contenant tous les titres ayant été dans le top 10 sur Spotify depuis 2013.')
st.write('On affiche la table ci dessous afin de déterminer la structure des colonnes et des données dans cette table.')

data.shape


st.write('Ensuite, on execute des commandes génériques afin de voir des données essencielles sur cette table.')

st.write("Description des données",data.describe())

st.write("Total de valeurs",data.count())

st.write('Afin d éviter les problèmes avec le nom des colonnes, on renomme la colonne "top genre" et supprime la colonne "Unnamed: 0", qui représente l index des données.')


data = data.rename(columns={'top genre': 'top_genre'})
data = data.drop('Unnamed: 0', axis=1)
st.write("Affichage des colonnes après suppression",data.columns)


st.write("Affichage des 5 premières valeurs",data.head(5))


st.write("Avant de savoir comment analyser les données de la table, nous regardons la table de correlation pour voir quelles données peuvent être liées dans la table.")

st.write(data.corr())

st.write("Pour mieux voir les résultats, on les représente avec des couleurs. On en profite aussi pour séparer les colonnes entre kes numériques et catégoriques pour pouvoir executer cette commande.")

fig, ax = plt.subplots()

categorical_cols = data.select_dtypes(include=['object']).columns
numerical_cols = data.select_dtypes(exclude=['object']).columns
nrows = data.shape[0]
ncols = data.shape[1]
sns.heatmap(data[numerical_cols].corr(), vmin = -1, vmax = 1, ax=ax)

st.write(fig)

st.write("Enfin, on ajoute les valeurs de la table de correlation pour plus de précision")
plt.clf()

fig, ax = plt.subplots()
plt.figure(figsize=(7, 6))
sns.heatmap(data[numerical_cols].corr(), annot = True, fmt = '.2f', cmap='Blues', vmin = -1, vmax = 1, ax=ax)
plt.title('Correlation entre les colonnes dans la table Spotify')

st.write(fig)


st.write("On remarque que les colonnes des paramètres ont donnés ces résultats:")

st.write(" - Acous lié à nrgy (-0.56)")

st.write(" - Val lié à dnce (0.50)")

st.write(" - dB lié à nrgy (0.54)")

st.write("On a ensuite représenté graphiquement ces paramètres entre eux:")

plt.clf()
fig, ax = plt.subplots()
data.plot(x='acous',y='nrgy',kind='scatter', title='Relation entre Energy et Acousticness',color='r', ax=ax)
plt.xlabel('Acousticness')
plt.ylabel('Energy')

st.write(fig)

plt.clf()
fig, ax = plt.subplots()
data.plot(x='nrgy',y='dB',kind='scatter', title='Relation entre Loudness (dB) et Energy',color='b', ax=ax)
plt.xlabel('Energy')
plt.ylabel('Loudness (dB)')
st.write(fig)

plt.clf()
fig, ax = plt.subplots()
data.plot(x='val',y='dnce',kind='scatter', title='Relation entre Loudness (dB) et Valence',color='g', ax=ax)
plt.xlabel('Valence')
plt.ylabel('Loudness (dB)')
st.write(fig)

st.write("On peut maintenant répondre à des problématiques intéressantes par rapport aux données trouvées.")

st.write("Par example, on peut trouver qui sont les artistes ayant le plus de chansons enregistrés dans la table de données")

artists = data['artist'].unique()
st.write("Il y a 184 artistes : ", len(artists))

artists = data['artist'].value_counts().reset_index().head(10)
st.write(artists)


st.write("Au vu des résultats de la commande précédente, on remarque que Katy Perry est l'artiste avec le plus de titres enregistrés dans le top 10 de Spotify entre 2010 et 2019")

st.write("Pour mieux visualiser ces résultats, on fait une représentation graphique:")


plt.clf()
fig, ax = plt.subplots()
plt.figure(figsize=(15,10))
sns.barplot(x='index',y='artist', data=artists, ax=ax)
plt.title("Nombre de titres en haut du classement des Top 10 artistes")
st.write(fig)


st.write("On peut aussi faire la séparation des titres de chaque artiste par année, comme dans la table ci dessous:")

t = []
topArtists = data['artist'].value_counts().head(10).index
for i in topArtists:
     t.append(data[data['artist'] == i])
        
resultArtist = pd.concat(t)
artistsYear = pd.crosstab(resultArtist["artist"],resultArtist["year"],margins=False)
st.write(artistsYear)


st.write("On remarque que l'artiste ayant eu l'année avec le plus de ses titres dans le top 10 de Spotify est Justin Bieber en 2015")

st.write("Une question que l'on peut se poser après voir ces résultats est: Quel a été la progression de chaque artiste au fil des années?")

st.write("Cette question est difficile à répondre sans représentation graphique, alors voici le graphique correspondant:")

plt.clf()
fig, ax = plt.subplots()
plt.figure(figsize=(20,10))
for i in artists['index']:
  tmp = []
  for y in range(2010,2020):
    songs = data[data['year'] == y][data['artist'] == i]
    tmp.append(songs.shape[0])
  sns.lineplot(x=list(range(2010,2020)),y=tmp, ax=ax)
plt.legend(list(artists['index']))
plt.title("Evolution de chaque artiste dans le Top 10 à travers les années")
st.write(fig)

plt.clf()
fig, ax = plt.subplots()
data['artist'].value_counts().head(10).plot.pie(figsize=(8,8), autopct='%1.0f%%')
st.write(fig)


st.write("De ces analyses, on découvre des données intéressantes sur les artistes en question:")

st.write("Justin Bieber par exemple, bien qu'il fasse partie du Top 10, n'a recu sa position que avec une montée impressionante de popularité entre 2014 et 2016.")

st.write("Similairement, Lady Gaga est une artiste intéressante: Elle a commencé en 2010-2011 avec entre 3 et 5 titres populaires, mais a baissé fortement jusqu'à 2018-2019, où elle est remontée en popularité. On remarque que cette remontée est fortement influencée par sa chanson dans le film 'A Star is Born' à la fin de 2018.")

st.write(data[data['artist'] == 'Lady Gaga'][data['year'] == 2018])


st.write("Une autre question interressante sur ces données serait de voir si des titres réapparaissent plus d'une fois dans le top 10?")

st.write("Tout d'abord, on regarde la liste des titres qui apparaissent plus d'une fois :")

st.write(data['title'].value_counts().head(20)>1)

st.write("Ensuite, on les affiche graphiquement :")


plt.clf()
fig, ax = plt.subplots()
plt.figure(figsize=(15,10))
sns.countplot(y=data.title, order=pd.value_counts(data.title).iloc[:19].index, data=data, ax=ax)
topMusics = data['title'].value_counts().head(19).index
plt.title("Titres apparaissant plus d'une fois")
st.write(fig)


st.write("On cherche ensuite la répartition de ces titres durant les années :")

plt.clf()
fig, ax = plt.subplots()
plt.figure(figsize=(20,10))
for i in topMusics:
  tmp = []
  for y in range(2010,2020):
    songs = data[data['year'] == y][data['title'] == i]
    tmp.append(songs.shape[0])
  sns.lineplot(x=list(range(2010,2020)),y=tmp, ax=ax)
plt.legend(list(topMusics))
plt.title("Evolution de chaque titre du top 10 répété plus d'une fois à travers les années")
st.write(fig)


st.write("On remarque que le titre 'Sugar' de Maroon 5 est revenu 2 fois durant la même année")

st.write(data[data['title']== 'Sugar'])

st.write("On a trouvé intéressant de voir quels sont les 15 titres les plus populaires dans la table de données")

st.write(data.sort_values(by=['pop'], ascending=False).head(15))

st.write("On découvre que la plupart des titres sont de 2019, avec une majorité de chansons du genre pop")

st.write("On retrouve en dessous les titres les plus longs, puis les titres avec la plus forte présence accoustique")

st.write(data.sort_values(by=['dur'], ascending=False).head(15))

st.write(data.sort_values(by=['acous'], ascending=False).head(15))

st.write("On en revient à une autre problématique intéressante:")

st.write("Lequel des 10 genres musicaux les plus connus dans la table de données est le plus populaire ?")

genres = data['top_genre'].value_counts().reset_index().head(10)

st.write(genres)

plt.clf()
fig, ax = plt.subplots()
plt.figure(figsize=(23,10))
sns.barplot(x='index',y='top_genre', data=genres, ax=ax)
st.write(fig)


st.write("On remarque que le genre de Dance Pop est de loin le plus populaire, avec plus de 300 titres.")

st.write("Pour comparaison, le 2eme genre le plus populaire est Pop, avec un peu plus de 50 titres.")

plt.clf()
fig, ax = plt.subplots()
data['top_genre'].value_counts().head(10).plot.pie(figsize=(8,8), autopct='%1.0f%%', ax=ax)
st.write(fig)

plt.clf()
fig, ax = plt.subplots()
plt.figure(figsize=(20,10))
for i in genres['index']:
  tmp = []
  for y in range(2010,2020):
    songs = data[data['year'] == y][data['top_genre'] == i]
    tmp.append(songs.shape[0])
  sns.lineplot(x=list(range(2010,2020)),y=tmp, ax=ax)
plt.legend(list(genres['index']))
st.write(fig)


st.write("On souhaite voir d'où viennent les artistes les plus populaires dans le top 10 des artistes de Spotify")

st.write(artists)

st.write("Vu que cette donnée ne vient pas de notre table, on entre les données manuellement")

dicArtists = {
    'Katy Perry':"Santa Barbara",
    'Justin Bieber':"London Canada",
     'Rihanna':"Saint Michael",
    'Maroon 5':"Los Angeles",
    'Lady Gaga':"Manhattan",
    'Bruno Mars':"Honolulu", 
    'The Chainsmokers':"Times Square" ,
    'Pitbull':"Miami",
    'Shawn Mendes':"Toronto",
    'Ed Sheeran':"United Kingdom", 
  }

st.write(dicArtists)

st.write("On définit les coordonnées en longitude et lattitude")

import geocoder
listGeo = []

for value in (dicArtists.values()):
    g = geocoder.arcgis(value)
    listGeo.append(g.latlng)

st.write(listGeo)

top_genres =[]
for key in (dicArtists.keys()):
    top_genres.append(data[data['artist']== key].top_genre.unique())

st.write(top_genres)

lat = []
log = []
for i in listGeo:
    lat.append(i[0])
    log.append(i[1])


colors = {
 'dance pop': 'pink',
 'pop': 'blue',
 'barbadian pop': 'green',
 'electropop': 'orange',
 'canadian pop': 'red',
}

st.write("Après les avoir définit, voila les coordonnées des artistes sélectionnés :")

dfLocation = pd.DataFrame(columns=['Name','Lat','Log','Gen'])
dfLocation['Name'] = artists['index']
dfLocation['Gen']  = np.array(top_genres)
dfLocation['Lat']  = lat
dfLocation['Log']  = log


st.write(dfLocation)

#st.write("On modélise une carte via folium (qui utilise Google Maps) pour représenter les zones correspondantes")

#plt.clf()
#fig, ax = plt.subplots()
#spotify = folium.Map(
#    location=[41.5503200,-8.4200500],
#    zoom_start=2
#)
#st.write(spotify)


#st.write("Enfin, on affiche les artistes dans leur coordonnées sur cette carte")


#for i in range(10):
#    singer = dfLocation.iloc[i]
#    folium.Marker(
#        
#        popup=singer['Name']+'-'+singer['Gen'],
#        location=[singer['Lat'], singer['Log']],
#    icon=folium.Icon(color=colors[singer['Gen']], icon='music')).add_to(spotify)
    
#st.write(spotify)


#st.write("En se servant de la heat map, on peut définir les zones qui concentrent le plus d'artistes :")

#spotify = folium.Map(
#    location=[41.5503200,-8.4200500],# Coordenadas retiradas do Google Maps
#    zoom_start=2
#)

#HeatMap(list(zip(lat, log))).add_to(spotify)
#st.write(spotify)










st.title("Prédiction du genre musical")


st.header('1°/ Chargement des données')

data_load_state = st.text('Chargement...')

music = pd.read_csv("music_genre.csv")
st.write(music.head(20))

data_load_state.text('Chargement...réussi!')


st.header('2°/ Exploration des données')

st.write(music.describe())

st.write(music.shape , 'Notre dataset contient donc 50005 lignes et 18 colonnes') 
st.write('Valeurs présentes dans nos données catégoriques')
st.write('obtained date :',music['obtained_date'].unique())
st.write('mode :',music['mode'].unique())
st.write('key :',music['key'].unique())
st.write('music genre :',music['music_genre'].unique())

st.write('II.1. Chercher et supprimer les doublons')

st.write('music.duplicated() ?', music.duplicated().any())
duplicated = music.duplicated()
st.write(music[duplicated])

st.write("Maintenant qu'on a trouvé les doublons et nous avons vérifier que c'est bien les seuls en regardant la valeurs avant et après, nous les supprimons car ils ne contiennent que des valeurs NaN")
music.drop([10000, 10001, 10002, 10003, 10004], inplace = True)
st.write('Le dataset contient désormais exactement 50000 instances.', music.shape)

music.reset_index(inplace = True)

music = music.drop(["index", "instance_id", "track_name", "obtained_date"], axis = 1)

st.write('II.3. Exploration des artistes')

st.write('Les observations qui ne contiennent pas de données sur les artites' , music[music["artist_name"] == "empty_field"] , "Nous avons 2489 lignes avec le nom d'artiste manquant")

artists = music["artist_name"].value_counts()[:20].sort_values(ascending = False)

st.write(artists)

st.bar_chart(data = artists , use_container_width=True)

music = music.drop(music[music["artist_name"] == "empty_field"].index)

top_20_artists = music["artist_name"].value_counts()[:20].sort_values(ascending = False)
st.write(top_20_artists)

st.bar_chart(data = top_20_artists , use_container_width=True)

st.write("Nous remarquons après cette visualisation qu'un bon nombre d'artiste présent dans le top 20 sont des artistes japonais, nous pouvons en conclure que les données ont été receuillies au Japon.")

st.write("Nous allons maintenant supprimer la colonnes des noms des artistes car nous n'avons pas besoin de cette information pour la prédiction du genre")

music.drop("artist_name", axis = 1, inplace = True)

st.header('3°/ Data visualisation')

st.write('Distribution des clés')
key = music['key'].value_counts().sort_values(ascending = False)
st.bar_chart(key)

st.write('Distribution des mode')
mode = music['mode'].value_counts().sort_values(ascending = False)
st.bar_chart(mode)

st.write('Distribution des genres de music')
music_genre = music['music_genre'].value_counts().sort_values(ascending = False)
st.bar_chart(music_genre)

music = music.drop(music[music["tempo"] == "?"].index)
music["tempo"] = music["tempo"].astype("float")
music["tempo"] = np.around(music["tempo"], decimals = 2)

numeric_features = music.drop(["key", "music_genre", "mode"], axis = 1)
st.write('Nos données numériques : ' , numeric_features)

st.header("4°/ Encodage des données catégorique")
st.write("La majorité des modèles machine learning ne supporte pas le type chaine de caractère ( string ) c'est pour cela que nous devons faire un encodage des données en utilisant labelEncoder.")

key_encoder = LabelEncoder()
music["key"] = key_encoder.fit_transform(music["key"])
st.write(music)

st.write(key_encoder.classes_)

mode_encoder = LabelEncoder()
music["mode"] = mode_encoder.fit_transform(music["mode"])
st.write(music)

st.write(mode_encoder.classes_)

st.subheader("Préprocessing des données")
st.write("Séparation de notre target ( music_genre ) du reste de nos données")

music_features = music.drop("music_genre", axis = 1)
music_labels = music["music_genre"]

st.write(music_features.head())

st.write(music_labels.head())

st.write("Ci dessous nous procédons au data scaling ( procéder qui consiste à restraindre nos données de 0 à 1 pour faciliter l'apprentissage ).")

scaler = StandardScaler()
music_features_scaled = scaler.fit_transform(music_features)
st.write(music_features_scaled.mean(), music_features_scaled.std())

st.write("Découpage nos données en train et test")

tr_val_f, test_features, tr_val_l, test_labels = train_test_split(
    music_features_scaled, music_labels, test_size = 0.1, stratify = music_labels)

train_features, val_features, train_labels, val_labels = train_test_split(
    tr_val_f, tr_val_l, test_size = len(test_labels), stratify = tr_val_l)

st.write(train_features.shape, train_labels.shape, val_features.shape, val_labels.shape, test_features.shape,   test_labels.shape)

st.header("Modélisation ( Machine Learning )")
st.write("Nous passons maintenant à la parite modélisation de nos données, on rappelle d'abord le but principale : Essayer de prédire le genre musical d'une chanson à partir de différentes données sur cette dernière.")

st.write("Pour cela nous avons opté pour le modèle Random Forest car c'est un modèle très utilisé et très efficace pour ce genre de classification.")

st.write("There are various classification algorithms. Random Forest is quite a common one. Its success, i.e., how well it will be capable to predict a label, will be measured with several success metrics. The most popular ones are \"accuracy\" and \"f1 score\". The former shows the ratio between true predictions and all samples ( 𝑎𝑐𝑐𝑢𝑟𝑎𝑐𝑦=𝑇𝑃+𝑇𝑁𝑇𝑃+𝑇𝑁+𝐹𝑃+𝐹𝑁 ). \"f1 score\" is the harmonic mean between \"precision\" ( 𝑇𝑃𝑇𝑃+𝐹𝑃 ) and \"recall\" ( 𝑇𝑃𝑇𝑃+𝐹𝑁 ). It is computed with the following formula: 𝑓1𝑠𝑐𝑜𝑟𝑒=2𝑇𝑃2𝑇𝑃+𝐹𝑃+𝐹𝑁 .")

st.write("GridSearchCV() is used to find a combination of hyper-parameters that returns the highest success metrics. It requires specifying one success metric; it is better to use \"f1_score\" instead the default one (\"accuracy\"). In general, \"accuracy\" is preferred when the class distribution is similar, while \"f1_score\" is used for imbalanced classes. Despite the similar distribution of genres in the music dataset, \"f1_score\" is a little bit more reliable. It should be instantiated with make_scorer before being passed to GridSearch().")

f1 = make_scorer(f1_score, average = "weighted")

st.write("RandomForestClassifier() has many tunable hyper-parameters but the most important ones are the number of estimators (i.e., the number of trees that are going to find the relationships between data points), and their maximum depth - the number of levels with nodes where computations happen. Another appropriate hyper-parameter is the number of samples per leaf, which specifies the minimum number of samples required to be at a leaf node. A range of values per hyper-parameter is defined in a dictionary. It is passed to the searching algorithm, along with the number of folds for cross-validation.")

params = {
    "n_estimators": [5, 10, 15, 20, 25],
    "max_depth": [5, 10, 15, 20, 25],
    "min_samples_leaf": [1, 2, 3, 4, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid = params, scoring = f1, cv = 5)

st.write('The instantiated search algorithm gets the training data. It will be used for finding the combination of hyper-parameters that returns the highest "f1_score".')

st.write(grid_search.fit(train_features, train_labels))

st.write(grid_search.best_params_)

model = RandomForestClassifier(n_estimators = 35, max_depth = 15, min_samples_leaf = 4)

st.write(model.fit(train_features, train_labels))

def classification_task(estimator, features, labels):
    """
    Cette fonction évalue la classification en en faisant une prédiction ('predict') et une évaluation ('score') 
    du modèle RandomForest
    
    Arguments: 
        Estimator, features (X) and labels (y).
    
    Returns: 
        La preformance du model calculé en fonction de l'accuracy et f1_score.
    """
    predictions = estimator.predict(features)
    
    print(f"Accuracy: {accuracy_score(labels, predictions)}")
    print(f"F1 score: {f1_score(labels, predictions, average = 'weighted')}")


st.header(classification_task(model, train_features, train_labels))


plt.figure(figsize = (8, 6))
sns.heatmap(confusion_matrix(train_labels, model.predict(train_features)),
    annot = True,
    fmt = ".0f",
    cmap = "vlag",
    linewidths = 2,
    linecolor = "red",
    xticklabels = model.classes_,
    yticklabels = model.classes_)
plt.title("Valeurs réels")
plt.ylabel("Valeurs prédites")
plt.tight_layout()
plt.show()


st.subheader("VII.4. L'importance des features")
st.write('Dans cette petite et dernière partie nous allons utiliser la fonction "feature_importances_" qui nous apporte une idée sur les paramètres que les plus important pour la prédiction.')

st.subheader("Nous remarquons que la popularité ( première feature ) est la plus importante avec un taux de 0.24 suivi de acousticness avec 0.98")

st.write(model.feature_importances_)