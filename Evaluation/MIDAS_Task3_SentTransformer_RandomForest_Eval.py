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

dfTest=df.sample(frac=0.2).reset_index(drop=True)

sentenceTransformerModel=SentenceTransformer('stsb-distilbert-base')

X=sentenceTransformerModel.encode(dfTest['description'])
y=dfTest['Label']

randomForestModel=joblib.load("Models/RandomForest.sav")

y_preds=randomForestModel.predict(X)

cmVal=ConfusionMatrix(actual_vector=list(y),predict_vector=list(y_preds))
print("Validation Evaluation Metrics:-")
print("Accuracy:-", cmVal.Overall_ACC*100)
print("F1 Micro:-", cmVal.F1_Micro)
print("Kappa:-",cmVal.Kappa)
print("MCC:-",cmVal.Overall_MCC)
