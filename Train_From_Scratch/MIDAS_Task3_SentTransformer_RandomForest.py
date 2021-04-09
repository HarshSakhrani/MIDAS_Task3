import joblib
import sys
from glob import glob  
import math
import shutil
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycm import *
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv("Datasets/Task3_Cleaned_TextModality.csv",index_col=0)

sentenceTransformerModel=SentenceTransformer('stsb-distilbert-base')
X = sentenceTransformerModel.encode(df['description'])
y = df['Label']

trainRatio=0.8
valRatio=0.1
testRatio=0.1

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1-trainRatio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=testRatio/(testRatio + valRatio)) 

randomForestModel=RandomForestClassifier()

randomForestModel.fit(x_train,y_train)

y_val_preds=randomForestModel.predict(x_val)
y_test_preds=randomForestModel.predict(x_test)

cmVal=ConfusionMatrix(actual_vector=list(y_val),predict_vector=list(y_val_preds))
print("Validation Evaluation Metrics:-")
print("Accuracy:-", cmVal.Overall_ACC*100)
print("F1 Micro:-", cmVal.F1_Micro)
print("Kappa:-",cmVal.Kappa)
print("MCC:-",cmVal.Overall_MCC)

cmTest=ConfusionMatrix(actual_vector=list(y_test),predict_vector=list(y_test_preds))
print("Test Evaluation Metrics:-")
print("Accuracy:-", cmTest.Overall_ACC*100)
print("F1 Micro:-", cmTest.F1_Micro)
print("Kappa:-",cmTest.Kappa)
print("MCC:-",cmTest.Overall_MCC)
