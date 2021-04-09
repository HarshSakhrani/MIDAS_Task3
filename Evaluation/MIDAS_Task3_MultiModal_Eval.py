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

df=pd.read_csv("Datasets/Task3_Cleaned_Multimodal.csv",index_col=0)

dfTest=df.sample(frac=0.2).reset_index(drop=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.data import Dataset, DataLoader

class FlipkartDataset(Dataset):

  def __init__(self,dataframe,bertTokenizer,maxLength,vision_transform,device):
    self.data=dataframe
    self.bertTokenizer=bertTokenizer
    self.maxLength=maxLength
    self.vision_transform=vision_transform
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    self.imgPath=str(self.data.iloc[idx,4])
    self.productDescription=str(self.data.iloc[idx,0])
    self.label=self.data.iloc[idx,3]

    self.image=Image.open(self.imgPath).convert('RGB')
    self.image=self.vision_transform(self.image)

    self.encodedInput=self.bertTokenizer.encode_plus(text=self.productDescription,
							padding='max_length',
							truncation="longest_first",
							max_length=self.maxLength,
							return_tensors='pt',
							return_attention_mask=True,
							return_token_type_ids=True).to(device)

    return self.image,self.encodedInput,self.label

preprocess = torchvision.transforms.Compose([
  torchvision.transforms.Resize((224,224)),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
flipkartTestDataset=FlipkartDataset(dataframe=dfTest,bertTokenizer=tokenizer,maxLength=128,device=device,vision_transform=preprocess)
testLoader=torch.utils.data.DataLoader(flipkartTestDataset,batch_size=8,shuffle=True)

class MultiModalNetwork(nn.Module):
  def __init__(self,preTrainedBert,preTrainedVGG,textInChannels=1,embeddingDimension=768,numHeads=8,numEncoderLayers=3,numClasses=27):
    super(MultiModalNetwork,self).__init__()

    self.textInChannels=textInChannels
    self.embDim=embeddingDimension
    self.numHeads=numHeads
    self.numEncoderLayers=numEncoderLayers
    self.numClasses=numClasses

    self.bert=self.freezeBert(preTrainedBert)

    self.vgg13=self.freezeVGG(preTrainedVGG)

    self.encoderLayer=nn.TransformerEncoderLayer(d_model=self.embDim,nhead=self.numHeads)
    self.encoderBlock=nn.TransformerEncoder(self.encoderLayer,num_layers=self.numEncoderLayers)
    self.kimConv0=nn.Conv2d(in_channels=self.textInChannels,out_channels=100,kernel_size=(2,self.embDim))
    self.kimConv1=nn.Conv2d(in_channels=self.textInChannels,out_channels=100,kernel_size=(3,self.embDim))
    self.kimConv2=nn.Conv2d(in_channels=self.textInChannels,out_channels=100,kernel_size=(4,self.embDim))
    self.kimConv3=nn.Conv2d(in_channels=self.textInChannels,out_channels=100,kernel_size=(5,self.embDim))
    self.dropoutLayer=nn.Dropout(p=0.5)
    self.fc=nn.Linear(800,self.numClasses)

  def forward(self,textInput,imgInput):

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

    imgOutput=self.vgg13(imgInput)

    combinedOutput=torch.cat((kimOutput,imgOutput),1)

    output=self.fc(self.dropoutLayer(combinedOutput))

    return output

  def freezeVGG(self,originalPreTrainedVGG):
    count=0
    for param in originalPreTrainedVGG.features.parameters():
      if count<14:
        param.requires_grad=False
      count=count+1
    features=list(originalPreTrainedVGG.classifier.children())[:-3]
    features[3]=nn.Linear(4096,400)
    originalPreTrainedVGG.classifier = nn.Sequential(*features)
    return originalPreTrainedVGG

  def freezeBert(self,model):
    for params in model.parameters():
      params.requires_grad=False
    return model

bertModel = BertModel.from_pretrained('bert-base-uncased')
vggModel=models.vgg13(pretrained=True)
multiModalModel=MultiModalNetwork(preTrainedBert=bertModel,preTrainedVGG=vggModel)
multiModalModel.to(device)
if torch.cuda.is_available():
  multiModalModel.load_state_dict(torch.load("Models/MultiModalModel.pt"))
else:
  multiModalModel.load_state_dict(torch.load("Models/MultiModalModel.pt",map_location=torch.device('cpu')))

multiModalModel.eval()

def checkClassificationMetrics(loader,model):

  completeTargets=[]
  completePreds=[]

  correct=0
  total=0
  model.eval()

  with torch.no_grad():
    for imgData,textData,targets in loader:

      targets=targets.to(device=device)
      imgData=imgData.to(device)

      scores=model(textInput=textData,imgInput=imgData)
      
      _,predictions=scores.max(1)

      targets=targets.tolist()
      predictions=predictions.tolist()

      completeTargets.append(targets)
      completePreds.append(predictions)

    completeTargetsFlattened=[item for sublist in completeTargets for item in sublist]
    completePredsFlattened=[item for sublist in completePreds for item in sublist]

    cm = ConfusionMatrix(actual_vector=completeTargetsFlattened, predict_vector=completePredsFlattened)
    return cm

CM=checkClassificationMetrics(testLoader,multiModalModel)

print("Evaluation Metrics:-")
print("Accuracy:-", CM.Overall_ACC*100)
print("F1 Micro:-", CM.F1_Micro)
print("Kappa:-",CM.Kappa)
print("MCC:-",CM.Overall_MCC)


