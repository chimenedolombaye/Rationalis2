import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.express as px
# Charger les données
df = pd.read_csv("Financial_inclusion_dataset.csv")

# Afficher des informations générales sur l'ensemble de données
st.write("Informations générales sur l'ensemble de données :")
st.write(df.info())

# Création d'un nuage de points avec Plotly Express

fig = px.scatter(df, x='country', y='gender_of_respondent', color='country', title='Nuage de points')
# Affichage de la figure
st.plotly_chart(fig)

# Gestion des valeurs manquantes
df = df.dropna()

# Supprimer les doublons
df = df.drop_duplicates()

# Encodage des caractéristiques catégorielles

df_encoded = pd.get_dummies(df, columns=['country', 'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type', 'location_type', 'cellphone_access'])

# Diviser les données
X = df_encoded.drop(['bank_account', 'uniqueid'], axis=1)
y = df_encoded['bank_account']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Initialiser le modèle de régression logistique
model = LogisticRegression()

# Entraîner le modèle
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer la performance du modèle
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy}')

# Interface utilisateur Streamlit
st.title('Application de Prédiction')

# Ajouter des champs de saisie pour les fonctionnalités
feature1 = st.number_input('Entrez la première valeur')
feature2 = st.number_input('Entrez la deuxième valeur')
# Faire des prédictions
if st.button('Faire une prédiction'):
    # Créer un tableau de valeurs de fonctionnalités à partir des champs de saisie
    feature_values = [[feature1, feature2]]

    # Faire la prédiction
    prediction = model.predict(feature_values)

    # Afficher la prédiction
    st.write('Prédiction :', prediction)

