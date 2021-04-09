from torchvision import models
from pycm import *
from transformers import BertTokenizer, BertModel
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import pickle
import sys
from glob import glob  
import math
import shutil
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataset
import torch.utils.data.dataloader
import torchvision.transforms as visionTransforms
import PIL.Image as Image
from torchvision.transforms import ToTensor,ToPILImage

df=pd.read_csv("Datasets/Task3_Cleaned_TextModality.csv",index_col=0)

dfTest=df.sample(frac=0.2).reset_index(drop=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.data import Dataset, DataLoader

class FlipkartDataset(Dataset):

  def __init__(self,dataframe,bertTokenizer,maxLength,device):
    self.data=dataframe
    self.bertTokenizer=bertTokenizer
    self.maxLength=maxLength
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    self.productDescription=str(self.data.iloc[idx,0])
    self.label=self.data.iloc[idx,3]

    self.encodedInput=self.bertTokenizer.encode_plus(text=self.productDescription,
							padding='max_length',
							truncation="longest_first",
							max_length=self.maxLength,
							return_tensors='pt',
							return_attention_mask=True,
							return_token_type_ids=True).to(device)
    
    return self.encodedInput,self.label


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
flipkartTestDataset=FlipkartDataset(dataframe=dfTest,bertTokenizer=tokenizer,maxLength=128,device=device)
testLoader=torch.utils.data.DataLoader(flipkartTestDataset,batch_size=8,shuffle=True)

class Encoder_KimCNN(nn.Module):
  def __init__(self,preTrainedBert,inChannels=1,embeddingDimension=768,numHeads=8,numEncoderLayers=3,numClasses=27):
    super(Encoder_KimCNN,self).__init__()

    self.inChannels=inChannels
    self.embDim=embeddingDimension
    self.numHeads=numHeads
    self.numEncoderLayers=numEncoderLayers
    self.numClasses=numClasses

    self.bert=self.freezeBert(preTrainedBert)
    
    self.encoderLayer=nn.TransformerEncoderLayer(d_model=self.embDim,nhead=self.numHeads)
    self.encoderBlock=nn.TransformerEncoder(self.encoderLayer,num_layers=self.numEncoderLayers)
    self.kimConv0=nn.Conv2d(in_channels=self.inChannels,out_channels=100,kernel_size=(2,self.embDim))
    self.kimConv1=nn.Conv2d(in_channels=self.inChannels,out_channels=100,kernel_size=(3,self.embDim))
    self.kimConv2=nn.Conv2d(in_channels=self.inChannels,out_channels=100,kernel_size=(4,self.embDim))
    self.kimConv3=nn.Conv2d(in_channels=self.inChannels,out_channels=100,kernel_size=(5,self.embDim))
    self.dropoutLayer=nn.Dropout(p=0.5)
    self.fc=nn.Linear(400,self.numClasses)

  def forward(self,input):

    bertOutput=self.bert(input_ids=input['input_ids'].squeeze(dim=1),
			attention_mask=input['attention_mask'].squeeze(dim=1),
			token_type_ids=input['token_type_ids'].squeeze(dim=1)).last_hidden_state
    
    encoderInput=bertOutput.transpose(1,0) 
    encoderOutput=self.encoderBlock(encoderInput)
    encoderOutput=encoderOutput.transpose(1,0)
    
    kimInput=encoderOutput.unsqueeze(1)
    
    conv0_Output=F.relu(self.kimConv0(kimInput)).squeeze(3)
    conv1_Output=F.relu(self.kimConv1(kimInput)).squeeze(3)
    conv2_Output=F.relu(self.kimConv2(kimInput)).squeeze(3)
    conv3_Output=F.relu(self.kimConv3(kimInput)).squeeze(3)
    
    conv0_Output=F.max_pool1d(conv0_Output,conv0_Output.size(2))
    conv1_Output=F.max_pool1d(conv1_Output,conv1_Output.size(2))
    conv2_Output=F.max_pool1d(conv2_Output,conv2_Output.size(2))
    conv3_Output=F.max_pool1d(conv3_Output,conv3_Output.size(2))

    kimOutput=torch.cat((conv0_Output.squeeze(dim=2),conv1_Output.squeeze(dim=2),conv2_Output.squeeze(dim=2),conv3_Output.squeeze(dim=2)),dim=1)

    output=self.fc(self.dropoutLayer(kimOutput))

    return output

  def freezeBert(self,model):
    for params in model.parameters():
      params.requires_grad=False
    return model

model = BertModel.from_pretrained('bert-base-uncased')
encoder_kimcnn=Encoder_KimCNN(preTrainedBert=model)
encoder_kimcnn.to(device)
if torch.cuda.is_available():
  encoder_kimcnn.load_state_dict(torch.load("Models/Encoder_KimCNN.pt")) 
else:
  encoder_kimcnn.load_state_dict(torch.load("Models/Encoder_KimCNN.pt",map_location=torch.device('cpu')))

encoder_kimcnn.eval()

def checkClassificationMetrics(loader,model):

  completeTargets=[]
  completePreds=[]

  correct=0
  total=0
  model.eval()

  with torch.no_grad():
    for data,targets in loader:

      targets=targets.to(device=device)

      scores=model(data)
      _,predictions=scores.max(1)

      targets=targets.tolist()
      predictions=predictions.tolist()

      completeTargets.append(targets)
      completePreds.append(predictions)

    completeTargetsFlattened=[item for sublist in completeTargets for item in sublist]
    completePredsFlattened=[item for sublist in completePreds for item in sublist]

    cm = ConfusionMatrix(actual_vector=completeTargetsFlattened, predict_vector=completePredsFlattened)
    return cm

CM=checkClassificationMetrics(testLoader,encoder_kimcnn)

print("Evaluation Metrics:-")
print("Accuracy:-", CM.Overall_ACC*100)
print("F1 Micro:-", CM.F1_Micro)
print("Kappa:-",CM.Kappa)
print("MCC:-",CM.Overall_MCC)

