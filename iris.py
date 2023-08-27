#import pandas as pd
#import numpy as np
import pickle

#df = pd.read_csv('iris.data')

#X = np.array(df.iloc[:, 0:4])
#y = np.array(df.iloc[:, 4:])

#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#y = le.fit_transform(y.reshape(-1))

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#from sklearn.svm import SVC
#sv = SVC(kernel='linear').fit(X_train,y_train)


#pickle.dump(sv, open('iri.pkl', 'wb'))




AP = []
MN = []

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn import metrics
from sklearn.model_selection import cross_val_score


DF0 = pd.read_csv("Crop_recommendation.csv")

NF = [a for a in DF0.columns if DF0[a].dtypes != 'O']
print("Numerical Features Count {}".format(len(NF)))
# 'O' is python object == string

DF0.isnull().sum()*100/len(DF0)
 # percentage of values NULL in each coloum , ie data cleaning stuff.

def RandomSamplingImputation(DF0, variable):
    DF0[variable]=DF0[variable]
    random_sample=DF0[variable].dropna().sample(DF0[variable].isnull().sum(),random_state=0)
    random_sample.index=DF0[DF0[variable].isnull()].index
    DF0.loc[DF0[variable].isnull(),variable]=random_sample

DF = [a for a in NF if len(DF0[a].unique())<25]
CF = [a for a in NF if a not in DF]

DF0.isnull().sum()*100/len(DF0)

for a in CF:
    if(DF0[a].isnull().sum()*100/len(DF0))>0:
        DF0[a] =DF0[a].fillna(DF0[a].median())

DF0.isnull().sum()*100/len(DF0)

def Mode_Nan(DF0,variable):
    mode=DF0[variable].value_counts().index[0]
    DF0[variable].fillna(mode,inplace=True)

DF0.isnull().sum()*100/len(DF0)

Cf = [a for a in DF0.columns if a not in NF]
DF0.isnull().sum()*100/len(DF0)
DF0['label'].unique()
DF0['label'].value_counts()

a = DF0[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
Target = DF0['label']
Labels = DF0['label']

X_train, X_test, y_train, y_test = train_test_split(a, Target, test_size =0.2, random_state = 2)

RF=RandomForestClassifier()
RF.fit(X_train,y_train)

pickle.dump(RF, open('iril.pkl', 'wb'))
model = pickle.load(open('iril.pkl', 'rb'))

PRED7 = RF.predict(X_test)
x = metrics.accuracy_score(y_test,PRED7)
AP.append(x)
MN.append('RF')
print("RF Accuracy ==>: ", x)
print(classification_report(y_test,PRED7))

pickle.dump(RF, open('iril.pkl', 'wb'))
model = pickle.load(open('iril.pkl', 'rb'))

O = cross_val_score(RF,a,Target,cv=5)

q = np.array([[ 22, 67, 78 , 17.1 , 14.42 , 6.2 , 72.3 ]])

PRED00 = RF.predict(q)
print("Most suitable crop is ")
print(PRED00)