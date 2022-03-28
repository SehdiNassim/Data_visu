import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_roc_curve, mean_squared_error, r2_score, accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Global variables
dummification, outliers, valeurs_manquantes, normalisation = [False for i in range(4)]

st.title("Bienvenue")

siteHeader = st.container()
dataExploration = st.container()
preprocessing = st.container()
modelTraining = st.container()
evaluation = st.container()

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
        print(df.columns)
        # Preprocessing
        st.subheader('Preprocessing:')
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

with modelTraining:
    try:
        print(df.columns)
        # Machine Learning models
        st.title("Classification supervisée")

        model_selected = st.selectbox(
        'Quel modèle souhaitez-vous appliquer ?',
        ('-', 'Régression logistique', 'Random Forest', 'Réseau de neuronnes'))

        input_vars = st.multiselect("Features à prendre en compte", df.columns)
        categorical = False

        y = df[[output_var+'_ouput']]
        X = df[input_vars]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        predictions = []
        classifier = None

        for v in input_vars:
            if v in newdf2:
                categorical=True
        

        if not categorical:
            if str(model_selected) == 'Régression logistique':
                # Hyperparameters
                max_iter = st.selectbox("Nombre maximal d'itérations", options=[100, 200, 300], index=0)

                # Model fitting
                classifier = LogisticRegression(max_iter=max_iter)
                classifier.fit(x_train, y_train)

                predictions = classifier.predict(x_test)

            elif str(model_selected) == 'Random Forest':
                # Hyperparameters
                nb_trees = st.selectbox("Nombre d'arbres dans la foret ?", options=[100, 200, 300, 'No limit'], index=0)
                max_depth = st.slider("Profondeur max des arbres ?", min_value=10, max_value=100, value=30, step=10)

                # Model fitting
                classifier = RandomForestClassifier(max_depth=max_depth, n_estimators=nb_trees)
                classifier.fit(x_train, y_train)

                predictions = classifier.predict(x_test)
                #st.table(predictions[:10])
                #rounded_predictions = np.rint(predictions).astype(int)           

            elif str(model_selected) == 'Réseau de neuronnes':
                # Hyperparameters
                hidden_layer_sizes = st.slider("Nombre de couches ?", min_value=2, max_value=20, value=2, step=1)
                activation = st.selectbox("Fonction d'activation ?", options=["relu", "logistic", "tanh", 'identity'], index=0)
                learning_rate_init = st.selectbox("Learning rate initial ?", options=[0.001, 0.005, 0.01, 0.05, 0.1], index=0)
            
                # Model fitting
                classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, learning_rate_init=learning_rate_init)
                classifier.fit(x_train, y_train)

                predictions = classifier.predict(x_test)
        else:
            st.warning('Veuillez ne pas inclure de variables catégrorielles ou les dummifier avant deffecter une classification...')
    except:
        st.warning('A problem occured while doing the ML process')

with evaluation:
    try:
        print(df.columns)
        st.subheader("Evaluation")

        # Model quality
        if predictions != [] :
            mse = mean_squared_error(y_test, predictions)
            acc = accuracy_score(y_test, predictions)
            rs = recall_score(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            fs = f1_score(y_test, predictions)
            cm = confusion_matrix(y_test, predictions)

            fig4, ax4 = plt.subplots()
            ax4.set_title("Confusion Matrix \n")
            sns.set(font_scale=1.4)
            sns.heatmap(cm, annot=True, annot_kws={'size':10}, cmap=plt.cm.Greens, linewidths=0.2)
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            st.pyplot(fig4)

            st.write("Tableau des métriques")
            eval_dict = {
                'Precision': acc,
                'Recall': rs,
                'F-score': fs,
                'MSE': mse,
                'Rsquared': r2
            }
            st.table(pd.DataFrame([eval_dict]))

            st.write("Courbes évaluatives")
            fig5, ax5 = plt.subplots()
            ax5.set_title("ROC Curve \n")
            plot_roc_curve(classifier, x_test, y_test, ax=ax5)
            st.pyplot(fig5)

            # Variables importantes
            st.subheader("Les features les plus importantes")
            pca = PCA(n_components=2)

            x_new = pd.DataFrame(data = pca.fit_transform(X), columns = ['PC1', 'PC2'])
            x_new["outcome"] = y

            fig6, ax6 = plt.subplots()
            sns.lmplot(x="PC1", y="PC2", data=x_new, hue="outcome")
            st.pyplot(sns.lmplot(x="PC1", y="PC2", data=x_new, hue="outcome"))

            most_important = [np.abs(pca.components_[i]).argmax() for i in range(2)]
            most_important_names = [input_vars[most_important[i]] for i in range(2)]
            dic = {'Variable qui contribue le plus à Axe {}'.format(i+1): most_important_names[i] for i in range(2)}
            adf = pd.DataFrame(dic.items())

            st.table(adf)
    except:
        st.warning('A problem occured during evaluation')
        

    