import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import pickle

dataset=pd.read_excel('19MAY_DATA_UPDATE.xlsx',delimiter=';')

X=dataset.iloc[:,:6].values
y=dataset.iloc[:,[6]].values


def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=0)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


pickle.dump(lm, open('reg_model.pkl','wb'))
model = pickle.load(open('reg_model.pkl','rb'))