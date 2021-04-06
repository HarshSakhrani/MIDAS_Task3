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

df=pd.read_csv("Task3_Cleaned_TextModality.csv",index_col=0)

dfTrain,dfVal,dfTest=np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df)), int(.9 * len(df))])
dfTrain=dfTrain.reset_index(drop=True)
dfTest=dfTest.reset_index(drop=True)
dfVal=dfVal.reset_index(drop=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.data import WeightedRandomSampler
freqLabels=torch.tensor(dfTrain['Label'].value_counts().sort_index(),dtype=torch.double)
weightClass=freqLabels/freqLabels.sum()
weightClass= 1/weightClass
weightClass=(weightClass).tolist()
sampleWeights=[weightClass[i] for i in dfTrain['Label']]
trainSampler=WeightedRandomSampler(sampleWeights,len(dfTrain))

from torch.utils.data import Dataset, DataLoader

class FlipkartDataset(Dataset):

  def __init__(self,dataframe,preTrainedBert,bertTokenizer,maxLength,device):
    self.data=dataframe
    self.bertTokenizer=bertTokenizer
    self.model=preTrainedBert 
    self.maxLength=maxLength
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    self.productDescription=str(self.data.iloc[idx,0])
    self.label=self.data.iloc[idx,3]

    					self.encodedInput=self.bertTokenizer.encode_plus(text=self.productDescription,padding='max_length',truncation="longest_first",max_length=self.maxLength,return_tensors='pt',return_attention_mask=True,return_token_type_ids=True).to(device)
    #print(self.encodedInput)
    self.model.to(device)
    			    self.embedding=self.model(input_ids=self.encodedInput['input_ids'],attention_mask=self.encodedInput['attention_mask'],token_type_ids=self.encodedInput['token_type_ids']).last_hidden_state

    return self.embedding,self.label


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

flipkartTrainDataset=FlipkartDataset(dataframe=dfTrain,preTrainedBert=model,bertTokenizer=tokenizer,maxLength=512,device=device)
flipkartTestDataset=FlipkartDataset(dataframe=dfTest,preTrainedBert=model,bertTokenizer=tokenizer,maxLength=512,device=device)
flipkartValDataset=FlipkartDataset(dataframe=dfVal,preTrainedBert=model,bertTokenizer=tokenizer,maxLength=512,device=device)

trainLoader=torch.utils.data.DataLoader(flipkartTrainDataset,batch_size=8,sampler=trainSampler)
testLoader=torch.utils.data.DataLoader(flipkartTestDataset,batch_size=8,shuffle=True)
valLoader=torch.utils.data.DataLoader(flipkartValDataset,batch_size=8,shuffle=True)


class Encoder_KimCNN(nn.Module):
  def __init__(self,inChannels=1,embeddingDimension=768,numHeads=8,numEncoderLayers=3,numClasses=27):
    super(Encoder_KimCNN,self).__init__()
    self.inChannels=inChannels
    self.embDim=embeddingDimension
    self.numHeads=numHeads
    self.numEncoderLayers=numEncoderLayers
    self.numClasses=numClasses

    self.encoderLayer=nn.TransformerEncoderLayer(d_model=self.embDim,nhead=self.numHeads)
    self.encoderBlock=nn.TransformerEncoder(self.encoderLayer,num_layers=self.numEncoderLayers)
    self.kimConv0=nn.Conv2d(in_channels=self.inChannels,out_channels=100,kernel_size=(2,self.embDim))
    self.kimConv1=nn.Conv2d(in_channels=self.inChannels,out_channels=100,kernel_size=(3,self.embDim))
    self.kimConv2=nn.Conv2d(in_channels=self.inChannels,out_channels=100,kernel_size=(4,self.embDim))
    self.kimConv3=nn.Conv2d(in_channels=self.inChannels,out_channels=100,kernel_size=(5,self.embDim))
    self.dropoutLayer=nn.Dropout(p=0.5)
    self.fc=nn.Linear(400,self.numClasses)

  def forward(self,input):
    input=input.squeeze(dim=1)
    input=input.transpose(1,0)
    
    encoderOutput=self.encoderBlock(input)
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

encoder_kimcnn=Encoder_KimCNN()
encoder_kimcnn.to(device)
softmaxLoss = nn.CrossEntropyLoss()
optimizer = optim.Adam(encoder_kimcnn.parameters(), lr=0.0001)

def Average(lst): 
    return sum(lst) / len(lst) 

def train_model(model,patience,epochs):


  trainBatchCount=0
  testBatchCount=0

  avgTrainAcc=[]
  avgValidAcc=[]
  trainAcc=[]
  validAcc=[]
  trainLosses=[]
  validLosses=[]
  avgTrainLoss=[]
  avgValidLoss=[]


  for i in range(epochs):

    print("Epoch:",i)

    model.train()
    print("Training.....")
    for batch_idx,(data,targets) in enumerate(trainLoader):

      trainBatchCount=trainBatchCount+1

      #data=data.to(device)
      targets=targets.to(device)

      optimizer.zero_grad()

      scores=model(data)
       
      loss=softmaxLoss(scores,targets)

      loss.backward()

      optimizer.step()

      trainLosses.append(float(loss))

      
      correct=0
      total=0
      total=len(targets)


      predictions=torch.argmax(scores,dim=1)

      if i==0:
        print("Targets:-",targets)
        print("Predictions:-",predictions)


      correct = (predictions==targets).sum()
      acc=float((correct/float(total))*100)

      trainAcc.append(acc)

      if ((trainBatchCount%200)==0):

        print("Targets:-",targets)
        print("Predictions:-",predictions)

        print('Loss: {}  Accuracy: {} %'.format(loss.data, acc))

    model.eval()
    print("Validating.....")
    for data,targets in testLoader:

      testBatchCount=testBatchCount+1

      #data=data.to(device=device)
      targets=targets.to(device=device)

      scores=model(data)

      loss=softmaxLoss(scores,targets)

      validLosses.append(float(loss))

      testCorrect=0
      testTotal=0

      _,predictions=scores.max(1)

      testCorrect = (predictions==targets).sum()
      testTotal=predictions.size(0)

      testAcc=float((testCorrect/float(testTotal))*100)

      validAcc.append(testAcc)

      if ((testBatchCount%200)==0):

        print('Loss: {}  Accuracy: {} %'.format(float(loss), testAcc))
    

    trainLoss=Average(trainLosses)
    validLoss=Average(validLosses)
    avgTrainLoss.append(trainLoss)
    avgValidLoss.append(validLoss)
    tempTrainAcc=Average(trainAcc)
    tempTestAcc=Average(validAcc)
    avgTrainAcc.append(tempTrainAcc)
    avgValidAcc.append(tempTestAcc)

    print("Epoch Number:-",i,"  ","Training Loss:-"," ",trainLoss,"Validation Loss:-"," ",validLoss,"Training Acc:-"," ",tempTrainAcc,"Validation Acc:-"," ",tempTestAcc)

    trainAcc=[]
    ValidAcc=[]
    trainLosses=[]
    validLosses=[]

    #early_stopping(validLoss,model)

    #if early_stopping.early_stop:
      #print("Early Stopping")
      #break

  #model.load_state_dict(torch.load("checkpoint.pt"))

  return model,avgTrainLoss,avgValidLoss,avgTrainAcc,avgValidAcc


encoder_kimcnn,avgTrainLoss,avgValidLoss,avgTrainAcc,avgValidAcc=train_model(encoder_kimcnn,3,10)



loss_train = avgTrainLoss
loss_val = avgValidLoss
epochs = range(1,11)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and Validation loss(Pre-Training)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('preTrainingLoss.png', bbox_inches='tight')
plt.close()
#plt.show()

loss_train = avgTrainAcc
loss_val = avgValidAcc
epochs = range(1,11)
plt.plot(epochs, loss_train, 'g', label='Training Acc')
plt.plot(epochs, loss_val, 'b', label='Validation Acc')
plt.title('Training and Validation Accuracy(Pre-Training)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('preTrainingAcc.png', bbox_inches='tight')
plt.close()

def checkClassificationMetrics(loader,model):

  completeTargets=[]
  completePreds=[]

  correct=0
  total=0
  model.eval()

  with torch.no_grad():
    for data,targets in loader:
      data=data.to(device=device)
      targets=targets.to(device=device)

      scores=model(data)
      _,predictions=scores.max(1)

      targets=targets.tolist()
      predictions=predictions.tolist()

      completeTargets.append(targets)
      completePreds.append(predictions)
      #correct += (predictions==targets).sum()
      #total+=predictions.size(0)


    completeTargetsFlattened=[item for sublist in completeTargets for item in sublist]
    completePredsFlattened=[item for sublist in completePreds for item in sublist]

    cm = ConfusionMatrix(actual_vector=completeTargetsFlattened, predict_vector=completePredsFlattened)
    return cm


softmaxCM=checkClassificationMetrics(testLoader,encoder_kimcnn)


f=open("softmaxResults.txt","a")
f.write(str(softmaxCM))
f.close()







