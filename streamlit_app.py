import streamlit as st
from streamlit_shap import st_shap

import pandas as pd
import numpy as np
import io
import base64
import pickle
import sklearn
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import shap
shap.initjs() # for visualization

########################################################################################################################################################################
# Définition du main()
########################################################################################################################################################################

def main():
	st.sidebar.title("RainsBerry")
	#st.set_page_config(
	#page_title="RainsBerry - Météo",
	#page_icon="👋",
	#layout="wide",)
	Menu = st.sidebar.radio(
		"Menu",
		('Le Projet Météo', 'Dataset & PreProcessing','DataViz','Modelisations','Performances','Simulations','Clustering','Séries Temporelles','Deep Learning','Conclusion','Rapport'))
	if Menu == 'Le Projet Météo':
		from PIL import Image
		image = Image.open('images/RainsBerry_2.jpg')
		st.image(image,width=600,caption="")
		'''
		* Le projet présenté dans ce streamlit a été développé dans le cadre de la formation Data Scientist de Datascientest.com - Promotion Octobre 2021.
		* L'objectif premier de ce projet est de mettre en application les différents acquis de la formation sur la problématique de prévision météo et plus précisément de répondre à une question essentielle: va-t-il pleuvoir demain?
		'''
		st.image('images/Intro_météo.jpg',width=600,caption="")
		'''
		* En dehors d'intéresser particulièrement les fabricants de parapluie, on comprend aussi que cette question est essentielle que ce soit dans le domaine des loisirs (gestion des parcs d'attraction), de l'agriculture, du traffic routier, et bien d'autres sujets.
		* Le lien du repo github est disponible ici: https://github.com/DataScientest-Studio/RainsBerryPy.
		'''
	if Menu == 'Dataset & PreProcessing':
		PreProcessing()
	if Menu == 'DataViz':
		DataViz()
	if Menu == 'Modelisations':
		Modelisations()
	if Menu == 'Performances':
		Performances()
	if Menu == 'Simulations':
		simulation()
	if Menu == 'Clustering':
		clustering()
	if Menu == 'Rapport':
		rapport()
	st.sidebar.text("")
	st.sidebar.text("Projet DataScientest")
	st.sidebar.text("Promotion DataScientist Octobre 2021")
	st.sidebar.text("Lionel Bottan")
	st.sidebar.text("Julien Coquard")
	st.sidebar.text("Samuel Guérin")
	st.sidebar.write("[Lien du git](https://github.com/DataScientest-Studio/RainsBerryPy)")

########################################################################################################################################################################
# Définition de la partie Preprocessing
########################################################################################################################################################################
    
def PreProcessing():
    
    from PIL import Image
    
    st.header("Dataset & PreProcessing")
    '''
    ###Dataset
    '''
    st.subheader("Fichier source")
    image = Image.open('images/weatherAUS.jfif')
    st.image(image, caption='Relevé Météo en Australie')
    df=pd.read_csv('data/weatherAUS.csv') #Read our data dataset
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.write("Présentation du jeu de données : ") 
    st.text(s)
    
    st.subheader("Ajout de nouvelles données") 
    
    st.write("Principaux climats australiens") 
    image = Image.open('images/grd_climats.png')
    st.image(image, caption='Climats australiens')
   	
    st.write("Classification de Köppen") 
    image = Image.open('images/clim_koppen.png')
    st.image(image, caption='Climats - Classification de Koppen')
	
    df=pd.read_csv('data/climatsAUS_v2.csv') #Read our data dataset
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.write("Présentation du jeu de données : ") 
    st.text(s)
    
    st.write("Coordonnées GPS")     
    image = Image.open('images/GPS.jfif')
    st.image(image, caption='Coordonnées GPS')
    df=pd.read_csv('data/aus_town_gps.csv') #Read our data dataset
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.write("Présentation du jeu de données : ") 
    st.text(s)
    
    '''
    ###Preprocessing
    '''
	


########################################################################################################################################################################
# Définition de la partie DataViz
########################################################################################################################################################################
 
def DataViz():
    st.header("DataViz")
    if st.checkbox("Corrélations de la pluie du lendemain (RainTomorrow) et de  l'ensoleillement (Sunshine)"):
        st.image('images/Dataviz_corr.jpg')
        '''
        #### Observations :
        * L’analyse des corrélations nous montre que les liaisons entre les différents critères sont nombreuses.
        * Quelles sont les variables les plus corrélées à RainTomorrow ?
            * Ensoleillement : Sunshine
            * Humidité : 3pm et 9am
            * Couverture nuageuse : 3pm et 9am
            * Pluie du jour : RainToday
            * Pression atmosphérique : Pressure3pm et Pressure9am
        * L'ensoleillement (Sunshine) est corrélé à RainTomorow_num malgré presque 50% de valeurs manquantes pour cette variable. Quand on regarde les corrélations, on peut imaginer de traiter ces valeurs manquantes en régressant Sunshine sur les critères les plus corrélés, à savoir :
            * Couverture nuageuse : 3pm et 9am
            * Humidité : 3pm et 9am
            * Température : Temp3pm, MaxTemp, Temp9am
        '''       
    if st.checkbox("Cartographie"):
        st.image('images/Dataviz_carto.jpg')
        '''
        #### Observations : 
        * Les stations météo d'Australie sont regroupées en 4 climats différents :
            * méditerrannéen : stations du sud-ouest et du sud-centre
            * chaud_humide (tropical et subtropical humide) => côte est du pays
            * tempéré_froid (tempéré océanique + montagnard) => plutôt sud-est
            * sec (chaud et semi-aride, voire aride) => intérieur du pays
        * La distribution mensuelle des précipitations illustre bien les différences de climat (mousson estivale pour le climat tropical, hivernale pour le climat méditerranéen).
        * Pour les stations au climat sec, on observe 9% de jours de pluie alors que pour les autres on est aux alentours de 22, 23%.
        '''       
    if st.checkbox("Influence sur la pluie du lendemain"):
        st.image('images/Dataviz_influence.jpg')
        '''
        #### Constats :
        * La distribution des variables Sunshine et Humidity3pm est bien différente selon RainTomorrow.
        * Pour MinTemp, la distribution est relativement similaire.
        * Pour Rainfall et Evaporation, il faut appliquer la fonction log pour neutraliser l'influence des valeurs extrêmes. On voit aussi l'influence plus importante de Rainfall sur RainTomorrow (distribution différente).
        '''      
	
########################################################################################################################################################################
# Définition de la partie modélisation
########################################################################################################################################################################

def Modelisations():
    st.header("Modélisations")
    
    Menu_mod = st.sidebar.radio(
     "Menu Modélisations",
     ('Equilibrage des classes','Traitement des valeurs manquantes','Sélection de variables', 'Conclusion'))

    def Equilibrage():
        st.subheader("Équilibrage des classes")
        st.image('images/model_01_desequilibre.jpg')
        st.markdown("**Performances d'un modèle Random Forest sur le jeu de données complet :**")
        st.image('images/model_02_sans_equ.jpg')
        if st.checkbox("Après équilibrage"):
            st.image('images/model_03_avec_equ.jpg')
	    st.image('images/model_04_PrecRap.jpg')
		
	#if st.checkbox("Précision et Rappel"):
        if st.checkbox("Modification du seuil de décision"):
            st.image('images/model_05_seuils_proba.jpg')
            st.image('images/model_06_seuilmaxF1.jpg')
        
    def TraitementNA():
        st.subheader("Traitement des valeurs manquantes")
        st.image('images/model_07_proportionsNA.jpg')
        if st.checkbox("Scores"):
            st.markdown("**Scores en fonction du jeu de données :**")
            st.image('images/model_08_scores_JD.jpg')
        
    def SelectionVar():
        st.subheader("Sélection de variables")
        st.image('images/model_09_selectKBest.jpg') 
  
    def Conclusion():
        st.subheader("Conclusion")
        
         
    if Menu_mod == 'Equilibrage des classes':
        Equilibrage()
        
    if Menu_mod == 'Traitement des valeurs manquantes':
        TraitementNA()
        
    if Menu_mod == 'Sélection de variables':
        SelectionVar()
        
    if Menu_mod == 'Conclusion':
        Conclusion()

########################################################################################################################################################################
# Définition de la partie perfomance
########################################################################################################################################################################

def Performances():
    st.header("Performances des modèles testés")
    '''
    #### Les algorithmes suivants ont été testés en prenant en compte les résultats des analyses précédentes :
    * Rééquilibrage du jeu de données avec RandomUnderSampler. 
    * Conservation de toutes les variables prédictives.
    * Choix de l'algorithme sur le dataset sans les NA (données réelles)
    * En revanche, application possible sur les données interpolées ce qui aurait l'intérêt de pouvoir avoir des prédictions sur les observations qui ont des valeurs manquantes (par exemple, les stations  qui ne mesurent pas certains indicateurs). 

    #### Liste des algorithmes testés :
    * Arbre de décision
    * Boosting sur arbre de décision (Adaboost classifier)
    * Isolation Forest (détection d’anomalies) => non présenté car vraiment trop dégradé.
    * Régression logistique
    * SVM
    * KNN
    * Random Forest
    * Light GBM
    * Bagging Classifier
    * Stacking Classifier (avec les modèles préentrainés RandomForest, SVM et LogisticRegression)
	
    ##### Optimisation des modèles :
    * Une grille de recherche sur les hyperparamètres a été construite pour les modèles avec le choix de maximiser le f1 comme métrique de performance et 3 folds pour limiter le surapprentissage.

    ##### Choix du modèle :
    * Le modèle final sera choisi au regard de la courbe de ROC, de l'AUC globale et surtout des métriques f1_score, precision, rappel sur la classe à modéliser.

    ##### Définitions :
    * La precision correspond au taux de prédictions correctes parmi les prédictions positives. Elle mesure la capacité du modèle à ne pas faire d’erreur lors d’une prédiction positive.
    * Le recall correspond au taux d’individus positifs détectés par le modèle. Il mesure la capacité du modèle à détecter l’ensemble des individus positifs.
    * Le F1-score évalue la capacité d’un modèle de classification à prédire efficacement les individus positifs, en faisant un compromis entre la precision et le recall (moyenne harmonique).
    ''' 
    if st.checkbox("Courbe de ROC"):
        st.image('images/Perf_ROC.jpg')       
    if st.checkbox("Selon le seuil de détection"):
        st.image('images/Perf_seuils.jpg')
        st.image('images/Perf_seuils1.jpg')          
    if st.checkbox("Conclusion"):
        '''
        * La comparaison des algorithmes sur la courbe de ROC nous donne une liste de quatre algorithmes sensiblement plus performants que les autres :
            * la Random Forest
            * le Bagging
            * la XGBoost
            * la Light GBM
        
        * Les comparaisons sur le F1_score en choisissant différents seuils de probabilités (0.50, F1_max, recall=precision) vont nous conduite à préférer la XGBOOST qui est légèrement plus performante que la lightGBM sur le seuil "recall=precision".
        '''
        st.image('images/Perf_conclusion1.jpg')

########################################################################################################################################################################
# Définition de la partie simulation
########################################################################################################################################################################

def simulation():
    #Chargement du modele
    picklefile = open("modeles/xgboost.pkl", "rb")
    modele = pickle.load(picklefile)  

    #Definition des features
    features = ["RainToday_Num","Rain_J-1","Rain_J-2","MinTemp","MaxTemp","Sunshine","Evaporation",
        "Humidity3pm","Humidity9am","Pressure9am","Pressure3pm","Cloud3pm","Cloud9am", 
        "Wind9am_cos","Wind3pm_cos","WindGust_cos","Wind9am_sin","Wind3pm_sin","WindGust_sin", 
        "Mois","Clim_type_det"]
                
    st.markdown("# Simulation")

    st.subheader("Lecture des données")

    Data = st.selectbox("DataFrame: " , ["echantillon","Sydney","AliceSprings","Darwin","Perth","Hobart"])

    if ( Data == "echantillon"):
        df=pd.read_csv('data/echantillon.csv') #Read our data dataset
    if ( Data == "Sydney"):
        df=pd.read_csv('data/Sydney.csv') #Read our data dataset
    if ( Data == "AliceSprings"):
        df=pd.read_csv('data/AliceSprings.csv') #Read our data dataset
    if ( Data == "Darwin"):
        df=pd.read_csv('data/Darwin.csv') #Read our data dataset
    if ( Data == "Perth"):
        df=pd.read_csv('data/Perth.csv') #Read our data dataset
    if ( Data == "Hobart"):
        df=pd.read_csv('data/Hobart.csv') #Read our data dataset    

    st.write("Nombre de lignes : ", df.shape[0]) 
    st.write("Nombre de colonnes : ", df.shape[1]) 

    st.subheader("DataViz")

    DataViz = st.selectbox("Quelle Dataviz ? : " , ["Part jours de Pluie","Correlation","Analyse mensuelle","Impact de RainTomorrow"])

    if ( DataViz == "Part jours de Pluie"):
        #Part des jours de pluie
        fig = plt.figure(figsize=(3,3))
        x = df.RainTomorrow_Num.value_counts(normalize=True)
        colors = sns.color_palette('pastel')[0:5]
        labels = ['Pas de pluie', 'Pluie']
        plt.pie(x, labels = labels, colors = colors, autopct='%.0f%%')
        plt.title("Part des jours de pluie")
        st.write(fig)

    if ( DataViz == "Correlation"):
        fig, ax = plt.subplots(figsize=(15,6))
        ListeCrit = ["RainTomorrow_Num","MinTemp","MaxTemp","Sunshine","Evaporation","Humidity3pm"]
        sns.heatmap(df[ListeCrit].corr(), cmap="YlGnBu",annot=True,ax=ax)
        st.write(fig)

        fig = plt.figure( figsize= (20, 7) )
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        corr = df.corr()
        ax1.title.set_text('Correlations de RainTomorrow')
        temp = corr[["RainTomorrow_Num"]].loc[abs(corr["RainTomorrow_Num"]) > 0.2].sort_values(by="RainTomorrow_Num",ascending=False)
        sns.heatmap(temp, cmap="YlGnBu",annot=True,ax=ax1)
        ax2.title.set_text('Correlations de Sunshine')
        temp = corr[["Sunshine"]].loc[abs(corr["Sunshine"]) > 0.2].sort_values(by="Sunshine",ascending=False)
        sns.heatmap(temp , cmap="YlGnBu",annot=True,ax=ax2)
        st.write(fig)


    if ( DataViz == "Analyse mensuelle"):
        fig, ax = plt.subplots(figsize=(15,6))
        ax.title.set_text("Distribution mensuelle des pluies")
        sns.lineplot(ax=ax,data=df, x="Mois", y="Rainfall")
        st.write(fig)

    if ( DataViz == "Impact de RainTomorrow"):
        fig, ax = plt.subplots(figsize=(20,4))
        plt.subplot(131)
        sns.histplot(data=df, x="Sunshine",hue="RainTomorrow_Num",bins=20, multiple="layer", thresh=None)
        plt.subplot(132)
        sns.histplot(data=df, x="MinTemp",hue="RainTomorrow_Num",bins=20, thresh=None)
        plt.subplot(133)
        sns.histplot(data=df, x="Humidity3pm",hue="RainTomorrow_Num",bins=20)
        st.write(fig)

    st.subheader("Prédiction")

    if st.button("Predict"):  
        #Courbe de ROC
        probs = modele.predict_proba(df[features])
        y_test =  df["RainTomorrow_Num"]
        fpr, tpr, seuils = sklearn.metrics.roc_curve(y_test, probs[:,1], pos_label=1)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        fig = plt.figure(figsize=(15,6))
        plt.plot(fpr, tpr, color='purple',  linestyle='--', lw=1, label='Model (auc = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle=':', label='Aléatoire (auc = 0.5)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux faux positifs')
        plt.ylabel('Taux vrais positifs')
        plt.title('Courbe ROC')
        plt.legend(loc="lower right");
        st.pyplot(fig)
    
        #Graphe selon le seuil 
        precision, recall, thresholds = precision_recall_curve(y_test, probs[:, 1], pos_label=1)
        dfpr = pd.DataFrame(dict(precision=precision, recall=recall, threshold=[0] + list(thresholds)))
        dfpr['F1']= 2 * (dfpr.precision * dfpr.recall) / (dfpr.precision + dfpr.recall)
        dfrpr_maxF1 = dfpr[dfpr.F1 == dfpr.F1.max()].reset_index()
        Seuil = dfrpr_maxF1["threshold"].values[0]
        dfpr["Diff_Recall_Precision"] = np.abs(dfpr["recall"]-dfpr["precision"])
        dfrpr_MinDiff = dfpr[dfpr.Diff_Recall_Precision == dfpr.Diff_Recall_Precision.min()].reset_index()
        Seuil1 = dfrpr_MinDiff["threshold"].values[0]
    
        fig = plt.figure(figsize=(15,6))
        plt.plot(dfpr["threshold"], dfpr['precision'],label="precision")
        plt.plot(dfpr["threshold"], dfpr['recall'],label="recall")
        plt.plot(dfpr["threshold"], dfpr['F1'],label="F1")
        plt.axvline(x=0.50,color="gray",label="seuil à 0.50")
        plt.axvline(x=Seuil,color="red",label="seuil maximisant F1")
        plt.axvline(x=Seuil1,color="purple",label="seuil Recall=Precision")
        plt.title("Choix du seuil sur la classe à modéliser")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        st.pyplot(fig)
        #Matrice de confusion
        y_pred = np.where(probs[:,1] >= 0.50, 1, 0)    
        y_pred_best = np.where( probs[:,1] >= Seuil, 1, 0)
        y_pred_best1 = np.where( probs[:,1] >= Seuil1, 1, 0)
        st.text('Matrice de confusion seuil 0.50 :\n ' + classification_report(y_test, y_pred))
        st.text('Matrice de confusion seuil maximisant F1 :\n ' + classification_report(y_test, y_pred_best))
        st.text('Matrice de confusion seuil Recall=Precision :\n ' + classification_report(y_test, y_pred_best1))    
        fig = plt.figure(figsize=(15,6))
        cm = confusion_matrix(y_test, y_pred_best)
        ConfusionMatrixDisplay(cm).plot()
        st.pyplot(fig)
        #Predictions
        prediction = modele.predict(df[features])
        predDf = pd.DataFrame(prediction,columns=["prediction"])
        Sortie = pd.concat([df[["Date","Location","Climat_Koppen","Clim_type_det","RainTomorrow_Num"]],predDf],axis=1)
        #st.write(Sortie)

    #st.subheader("Interprétabilité")
    
    #if st.button("Importance des features"):
    #    picklefile = open("modeles/xgboost.pkl", "rb")
    #    modele = pickle.load(picklefile)  
    #    explainer = shap.TreeExplainer(modele)
    #    shap_values = explainer.shap_values(df[features])
    #    st_shap(shap.summary_plot(shap_values, df[features]),height=300)

########################################################################################################################################################################
# Définition de la partie rapport
########################################################################################################################################################################
    
def rapport():
    st.write("[Lien git_hut :](https://github.com/DataScientest-Studio/RainsBerryPy)")
    def show_pdf(file_path):
        with open(file_path,"rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

    show_pdf('https://github.com/SamuelGuerin-Git/RainsBerryPy_save/blob/cac5fac60f5e539aec938a343b8152b3587f9ba4/RainsberryPy%20Meteo%20-%20Rapport%20final.pdf')

########################################################################################################################################################################
# Définition de la partie clustering
########################################################################################################################################################################
    
def clustering():
 
    Menu_mod = st.sidebar.radio(
     "Menu Clustering",
     ('Introduction et stratégie','1ère étape: Type de climat','2ème étape: Régime pluviométrique','3ème étape: Variation de température', 'Conclusion'))  
    
    def Intro():
        st.subheader("Introduction")
        st.image('images/clustering-in-machine-learning.jpg')

        ''' 
        #### La classification de Köppen est une classification des climats fondée sur les précipitations et les températures. Un climat, selon cette classification, est repéré par un code de deux ou trois lettres :
        * 1ère lettre : type de climat 
        * 2ème lettre : régime pluviométrique 
        * 3ème lettre : variations de températures.
        #### La combinaison de ces sous-classifications donne la classification de climat de Köppen suivante :
        '''        
        
        st.image('images/Climat de Koppen.jpg',caption='Classification de Koppen')
        ''' 
        ##### Stratégie Adoptée :
        * 1ère lettre : type de climat => Algorithme KMeans
        * 2ème lettre : régime pluviométrique => TimeSeriesKmeans Clustering
        * 3ème lettre : variations de températures => TimeSeriesKmeans Clustering
        '''                
        
        
    def KMeans():
        st.subheader("Clustering: Type de climat => KMeans")
        '''
        ### Preprocessing:
        #### Création d'un dataframe avec :
        * une ligne par ville
        * pour chaque variable considérée, création d'un jeu de douze colonnes avec le calcul de la moyenne mensuelle: 
            * 'MinTemp','MaxTemp','Temp9am','Temp3pm',
            * 'Rainfall',
            * 'Evaporation',
            * 'Sunshine',
            * 'WindGustSpeed','WindSpeed9am','WindSpeed3pm',
            * 'Humidity9am','Humidity3pm',
            * 'Pressure9am','Pressure3pm',
            * 'Cloud9am','Cloud3pm',
            * 'RainToday_Num'
        ### Utilisation de l'algorithme KMeans:
        #### Méthode du coude pour définir le nombre de clusters
        '''
        st.image('images/1L_Coude.jpg')
        '''
        #### Nous considérons 10 clusters.
        
        ### Comparaison Classification de Koppen vs Clustering 
        '''
        st.image('images/1L_ResultatsTab.jpg')
        '''
        ### Comparaison localisée
        '''
        st.image('images/1L_ResultatsMap.jpg')
        '''
        #### => Climats extrêmes bien identifiés mais résultats moins convaincants pour les autres. 
        '''
    def TSClustering2L():
        st.subheader("Clustering: Régime pluviométrique => TimeSeriesKmeans")
        '''
        ### Preprocessing
        ##### Sélection d'une plage de 3 ans et demi de données à partir de janvier 2014 - Plus grand plages avec des relevés consécutifs (données d'origine avec traitement KNN imputer).

        #### Résultats du Clustering de Séries Temporelles:
        '''
        st.image('images/2L_ResultatsPlot.jpg')
        '''
        ### Comparaison Classification de Koppen vs Clustering
        '''
        st.image('images/21L_ResultatsTab.jpg')
        '''
        ### Comparaison Localisée
        '''        
        st.image('images/2L_ResultatsMap.jpg')
        '''
        ##### => Le régime de mousson est bien isolé et le régime f associé au climat humide se retrouve seul dans de nombreux clusters (hormis 1).
        '''
        
    def TSClustering3L():
        st.subheader("Clustering: Variation de température")
        '''
        ### Preprocessing
        ##### Similaire à la classification précédente

        #### Résultats du Clustering de Séries Temporelles:
        '''
        st.image('images/3L_ResultatsPlot.jpg')
        '''
        ### Comparaison Classification de Koppen vs Clustering
        '''
        st.image('images/3L_ResultatsTab.jpg')
        '''
        ### Comparaison Localisée
        '''
        st.image('images/3L_ResultatsMap.jpg')
        '''
        ##### => L’ensemble des classifications des variations de température est dans l’ensemble bien exécuté.
        '''
    def Conclusion(): 
        st.subheader("Conclusion")
        '''
        ### Combinaison des différents clusters:
        '''
        st.image('images/Clust_ResultatsTab.jpg')
        '''
        #### 32 clusters différents identifiés
        '''
        st.image('images/FinalClust_ResultatsTab.jpg')
        '''
        #### Après regroupement des clusters identifiés sous la même classification de Koppen:
        '''
        st.image('images/Final_ResultatsMap.jpg')
        
    if Menu_mod == 'Introduction et stratégie':
        Intro()
        
    if Menu_mod == '1ère étape: Type de climat':
        KMeans()
        
    if Menu_mod == '2ème étape: Régime pluviométrique':
        TSClustering2L()
        
    if Menu_mod == '3ème étape: Variation de température':
        TSClustering3L()
        
    if Menu_mod == 'Conclusion':
        Conclusion()
        
if __name__ == "__main__":
    main()
