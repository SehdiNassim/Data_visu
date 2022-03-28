import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

# Global variables
dummification, outliers, valeurs_manquantes, normalisation = [False for i in range(4)]


st.title("Bienvenue")

siteHeader = st.container()
dataExploration = st.container()
preprocessing = st.container()
modelTraining = st.container()

with siteHeader:
    filename = st.file_uploader('Enter a file path:')
    try:
        if filename is not None:
            input = pd.read_csv(filename)
            df = input.copy()

    except FileNotFoundError:
        st.error('File not found.')
with dataExploration:
    try:
        st.table(df.head())
        st.title("Analyse Exploratoire")
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        newdf = df.select_dtypes(include=numerics)
        newdf2 = df.drop(columns=newdf.columns)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nb_Ligne", df.shape[0])
        col1.metric("Nb valeurs manquantes", df.isna().sum().sum())
        col2.metric("Nb_colonne", df.shape[1])
        col3.metric("Quantitative features", len(newdf.columns))
        col4.metric("Qualitative features", len(newdf2.columns))


        # Analyse unidimensionnelle
        st.subheader("Analyse Exploratoire Univariée sur variable quantitative")
        col1, col2 = st.columns([1, 3])
        current_column = col1.radio("Veuillez selectionner un feature", newdf.columns)
        col2.table(newdf[current_column].describe())
        fig1, ax1 = plt.subplots()
        ax1.set_title('Boite à moustache de la variable {}'.format(current_column))
        ax1.boxplot(newdf[current_column])
        col2.pyplot(fig1)

        st.subheader("Analyse Exploratoire Univariée sur variable qualitative")
        col1, col2 = st.columns([1, 3])
        current_column2 = col1.radio("Veuillez selectionner un feature", newdf2.columns)
        # col2.table(newdf2[current_column2].value_counts())
        fig2, ax2 = plt.subplots()
        ax2.set_title('diagramme circulaire de la variable {}'.format(current_column2))
        value_counts=newdf2[current_column2].value_counts()
        ax2.pie(value_counts.values,labels=value_counts.index)
        col2.pyplot(fig2)
        # Analyse bidimensionnelle
        st.subheader("Analyse Exploratoire Bivariée")
        col1, col2 = st.columns(2)
        feature_x = col1.selectbox("Premiere variable", newdf.columns)
        feature_y = col2.selectbox("Deuxieme variable", newdf.drop(feature_x, axis=1).columns)
        st.bar_chart(newdf.corr()[feature_x].drop(feature_x))
        corr=newdf.corr()
        fig3, ax3 = plt.subplots()
        ax3.set_title("Matrice de corrélation \n")
        sns.heatmap(corr,cmap="Blues",annot=True, xticklabels=corr.columns, yticklabels=corr.columns, ax=ax3)
        st.pyplot(fig3)
    except:
        st.warning('No file yet to explore')
with preprocessing:
    try:
        # Preprocessing
        st.write('Preprocessing:')
        st.write('Quel est votre outcome ?')
        output_var = st.selectbox('', df.columns)
        df[output_var+'_ouput']= LabelEncoder().fit_transform(df[[output_var]])
        droped_vars = st.multiselect('Quelles colonnes voulez vous supprimer ?', df.columns)
        df = df.drop(droped_vars, axis=1)
        st.write('Quels prétraitements souhaitez vous appliquer à vos features ?')
        col1, col2 = st.columns(2)
        dummification = col1.checkbox('Dummification')
        outliers = col1.checkbox('Outliers')
        valeurs_manquantes = col2.checkbox('Valeurs manquante')
        normalisation = col2.checkbox('Normalisation')
        try:
            if dummification: df = pd.get_dummies(df, columns=newdf2.drop(output_var, axis=1).columns)
        except KeyError:
            if dummification: df = pd.get_dummies(df, columns=newdf2.columns)
        if valeurs_manquantes: df = df.dropna()
        st.table(df.head())
    except:
        st.warning('An error occured during preprocessing')