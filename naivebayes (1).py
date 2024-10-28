# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:37:32 2023

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


x=pd.read_csv('train (1).csv')
x.drop(x[(pd.isna(x.Age))].index,axis=0,inplace=True)
y=x[['Survived']]
x.drop(['PassengerId','Name','Survived','Cabin'],axis=1,inplace=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
x_train=x_train.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)


numeric_varibale=['Age','Fare']
def find_max_index(value):
    new_list=sorted(value)
    for i in range(0,len(value)):
         if value[i]==new_list[-1]:   
           return i

def calculate_numeric_pro(data,sample):
    mean=np.mean(data)
    std=np.std(data)
    result=(1/(std*np.sqrt(2*np.pi)))*(np.exp(-((np.square(sample-mean))/(2*np.square(std)))))
    return result
    
def find_class_pro(x_train,y_train,sample,numeric_varibale):
    value_of_class=np.array(y['Survived'].value_counts().index)
    first_class=x_train.iloc[y_train[y_train['Survived']==value_of_class[0]].index]
    second_class=x_train.iloc[y_train[y_train['Survived']==1].index]
    pro_class_1=[]
    pro_class=1
    for i in range(0,len(first_class.columns)):
        if first_class.columns[i] in numeric_varibale:
            pro_class*=calculate_numeric_pro(first_class.iloc[:,i].values,sample[i])
        else: 
            pro_class*=len(first_class[first_class.iloc[:,i].values==sample[i]])/len(first_class)   
    pro_class_1.append(pro_class*(len(first_class)/len(first_class)+len(second_class)))
    pro_class=1
    for i in range(0,len(second_class.columns)):
        if second_class.columns[i] in numeric_varibale:
            pro_class*=calculate_numeric_pro(second_class.iloc[:,i].values,sample[i])
        else:
             pro_class*=len(second_class[second_class.iloc[:,i].values==sample[i]])/len(second_class)
    pro_class_1.append(pro_class*(len(second_class)/len(first_class)+len(second_class)))
    result=find_max_index(pro_class_1)
    return result


def naive_bayes(x_train,y_train,x_test,y_test,numeric_varibale):
    predict_class=[]
    for i in range(0,len(x_test)):
        predict_class.append(find_class_pro(x_train,y_train,np.array(x_test.iloc[i]),numeric_varibale))
    print(accuracy_score(y_test, predict_class))


naive_bayes(x_train,y_train,x_test,y_test,numeric_varibale)