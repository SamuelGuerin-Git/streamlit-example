import streamlit as st
from streamlit_shap import st_shap

import pandas as pd
import numpy as np
import io
import pickle
import sklearn
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import shap
shap.initjs() # for visualization

def main():
    st.sidebar.title("RainsBerry")
    
    #st.set_page_config(
    #page_title="RainsBerry - Météo",
    #page_icon="👋",
    #layout="wide",)

    Menu = st.sidebar.radio(
     "Menu",
     ('Le Projet Météo', 'PreProcessing','DataViz','Modelisations','Performances','Simulations','Clustering','Séries Temporelles','Deep Learning','Conclusion'))

    if Menu == 'Le Projet Météo':
        from PIL import Image
        image = Image.open('images/RainsBerry.jpg')
        st.image(image, caption='RainsBerry')
    if Menu == 'PreProcessing':
        PreProcessing()
    if Menu == 'DataViz':
        DataViz()
    if Menu == 'Modelisations':
        Modelisations()       
    if Menu == 'Simulations':
        simulation()
    if Menu == 'Clsutering':
        clustering()

    st.sidebar.text("")
    st.sidebar.text("Projet DataScientest")
    st.sidebar.text("Promotion DataScientist Octobre 2021")
    st.sidebar.text("Lionel Bottan")  
    st.sidebar.text("Julien Coquard")  
    st.sidebar.text("Samuel Guérin")      
    
def PreProcessing():
    
    from PIL import Image
    
    st.header("PreProcessing")
    
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
    
    st.write("Classification de Koppen") 
    image = Image.open('images/Climats.jpg')
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
    
    
    
    
    
    
def DataViz():
    st.header("DataViz")

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

    st.subheader("Interprétabilité")
    
    #if st.button("Importance des features"):
    #    picklefile = open("modeles/xgboost.pkl", "rb")
    #    modele = pickle.load(picklefile)  
    #    explainer = shap.TreeExplainer(modele)
    #    shap_values = explainer.shap_values(df[features])
    #    st_shap(shap.summary_plot(shap_values, df[features]),height=300)
    
def clustering():
 
    Menu_mod = st.sidebar.radio(
     "Menu Clustering",
     ('Introduction et stratégie','1ère étape: Type de climat','2ème étape: Régime pluviométrique','3ème étape: Variation de température', 'Conclusion'))  
    
    def Intro():
        st.subheader("Introduction")
        image = Image.open('images/weatherAUS.jfif')       
    def KMeans():
        st.subheader("Clustering: Type de climat")
        
    def TSClustering2L():
        st.subheader("Clustering: Régime pluviométrique")
        
    def TSClustering3L():
        st.subheader("Clustering: Variation de température")
        
    def Conclusion(): 
        st.subheader("Conclusion")

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
