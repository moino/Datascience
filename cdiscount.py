# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
#%%
import os

model = RandomForestClassifier(n_estimators=200,n_jobs=3)
row = 1000000
word = 5000

t = timer()
path = "/Users/admin/Documents/data_science/datascience/cdiscount/"

'''chargement CSV'''
df_rayon=pd.read_csv(path+"rayon.csv",delimiter=';')
df_train=pd.read_csv(path+"training.csv",delimiter=';',nrows = row)
df_test=pd.read_csv(path+"test.csv",delimiter=';',nrows=10)

print 'chargement csv done'
"traitement de data sur le fichier Train et test"

"""
di_rayon1={}
di_rayon2={}
di_rayon3={}

for j in range(1,5794) :
    di_rayon1.update({df_rayon.Categorie1[j] : df_rayon.Categorie1_Name[j]})
di_rayon2={}
for j in range(1,5794) :
    di_rayon2.update({df_rayon.Categorie2[j] : df_rayon.Categorie2_Name[j]})
for j in range(1,5794) :
    di_rayon3.update({df_rayon.Categorie3[j] : df_rayon.Categorie3_Name[j]})

df_train['Categorie1'] = df_train['Categorie1'].map(di_rayon1)
df_train['Categorie2'] = df_train['Categorie2'].map(di_rayon2)
df_train['Categorie3'] = df_train['Categorie3'].map(di_rayon3)

print 'demarrage analyse classe'


nb_sample = df_train['Categorie1'].shape

"analyse repartition classe"
'df_train['count'] = 1

K1 = df_train.groupby(df_train['Categorie1']).sum()
K1 = K1.sort('count')
K1['Frequence']=K1['count']/nb_sample
K1['Frequence'][K1['Frequence']>0.01].plot(kind='bar')

K1 = df_train.groupby(df_train['Categorie2']).sum()
K1 = K1.sort('count')
K1['Frequence']=K1['count']/nb_sample
K1['Frequence'][K1['Frequence']>0.01].plot(kind='bar')
K1 = df_train.groupby(df_train['Categorie3']).sum()
K1 = K1.sort('count')
K1['Frequence']=K1['count']/nb_sample
K1['Frequence'][K1['Frequence']>0.01].plot(kind='bar')
"""

df_train['text']=df_train['Description']+df_train['Libelle']
df_test['text']=df_test['Description']+df_test['Libelle']

ident_train= df_train['Identifiant_Produit']
ident_test = df_test['Identifiant_Produit']
df_train = df_train.drop(['Identifiant_Produit'],axis=1)
df_test = df_test.drop(['Identifiant_Produit'],axis=1)
df_train = df_train.drop(['Description','Libelle','Marque','Produit_Cdiscount'],axis=1)
df_test = df_test.drop(['Description','Libelle','Marque'],axis=1)
df_train = df_train.dropna()

print 'creation bag of word...'


from stop_words import get_stop_words
stop_wordsfr=get_stop_words('french')
tfv = TfidfVectorizer(min_df=3,  max_features= word, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = stop_wordsfr)

'''traintext = list(df_train.apply(lambda x:'%s %s %s' % (df_train['text']),axis=1))'''
#faire une analyse si on peut pas mieux fitter le sample test pour 
#faire une colonne somme de texte libellé plus description

tfv.fit(df_train['text'])
text_features = tfv.get_feature_names()
train_txt =  tfv.transform(df_train['text']).toarray()
test_txt = tfv.transform(df_test['text']).toarray()



df_train_txt=pd.DataFrame(train_txt,columns=text_features)
df_test_txt=pd.DataFrame(test_txt,columns=text_features)

print "bag of word ready"

y=df_train.Categorie3
y2=df_train.Categorie2
y1=df_train.Categorie1

"""
df_train['prix']=(df_train['prix']-df_train['prix'].mean())/(df_train['prix'].max()-df_train['prix'].min())
"""
print "model fitting..."

df_train,df_test1,y_train, y_test1  = train_test_split(df_train_txt,y,test_size=0.2,random_state=42)

model.fit(df_train, y_train)
print "model fitted"
'''Cross val score'''
y_pred=model.predict(df_test1)

cv = "Cross validation ="+ repr(round(accuracy_score(y_test1,y_pred),3))

'''os.system("say Result Ready")'''
elapsed_time = timer() - t 
duree =round(elapsed_time/60,2)
t = "durée =" + repr(duree) + " min"
print t
print cv

#%% 'submission
df_subm=pd.DataFrame()
df_subm.index=df_sample.index
df_subm.columns=df_sample.columns

df_subm.to_csv(path_or_buf =path+"subm.csv",header=True,index=True)

#inclure marque dans nom 
#supprimer ifll na marque mais faire attention merde avec NAN marque
#preparer le jeu de test bag of word
#mesure cross val
#faire une fonction bag of word

