# @Time : 2022/4/28 10:58
# @Author : hongzt
# @File : rgb-cnn
import  torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import warnings
warnings.filterwarnings("ignore")
from torch.autograd import variable
import torch.nn.functional as F
import  matplotlib.pyplot as plt
import pandas as pd
srcfst_train=pd.read_csv("F:\\database\\srcfst\\srcfstmax\\train.csv")
srcfst_test=pd.read_csv("F:\\database\\srcfst\\srcfstmax\\test.csv")
srcfst_train=pd.get_dummies(srcfst_train)#字符串独热编码
t_t_value=srcfst_train[["D","T","L","长细比","fty","As","fsy","fc","边界条件_平板支座","边界条件_平板铰支座"]].values
t_t_label=srcfst_train[["N"]].values

t_ftrain_tensor=torch.from_numpy(t_t_value).float()
t_ltrain_tensor=torch.from_numpy(t_t_label).float()
v_f=variable(t_ftrain_tensor)
v_l=variable(t_ltrain_tensor)
train_datase=TensorDataset(t_ftrain_tensor,t_ltrain_tensor)
train_loader=DataLoader(dataset=train_datase,batch_size=52,shuffle=True,num_workers=2)

#v_f=variable(train_loader)


#V_l=variable(t_ltrain_tensor)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1=nn.Sequential(#1,10
            nn.Conv1d(in_channels=1,out_channels=10,kernel_size=1,stride=2),
            nn.ReLU(),#1,10

            nn.MaxPool1d(2)#10,5
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(in_channels=10,out_channels=16,kernel_size=1,stride=2),#10,5
            nn.ReLU(),#16,5
            nn.MaxPool1d(1))#16,5
        self.out=nn.Linear(16*5,1)
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        output=self.out(x)
        return output
    
cnn=CNN()
optimizer=torch.optim.Adam(cnn.parameters(),lr=0.05)
loss_func=torch.nn.MSELoss()
for epoch in range(200):
    for i,(input,labels) in enumerate(train_loader,0):
        predic = cnn(input)
        loss = loss_func(predic, labels)

        print(loss)
        optimizer.zero_grad()  # tidu重新置0
        loss.backward()
        optimizer.step()
import keras
model.compile(optimizer=keras.)
