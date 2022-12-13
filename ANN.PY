# @Time : 2022/4/9 23:08
# @Author : hongzt
# @File : ann
import  torch
from torch.autograd import variable
import torch.nn.functional as F
import  matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#导入数据
#导入数据srcfst注意：此处不能使用汉语的路径
srcfst_train=pd.read_csv("F:\\database\\srcfst\\srcfstmax\\train.csv")
srcfst_test=pd.read_csv("F:\\database\\srcfst\\srcfstmax\\test.csv")
srcfst_train=pd.get_dummies(srcfst_train)#字符串独热编码
#调用值


t_t_value=srcfst_train[["D","T","L","长细比","fty","As","fsy","fc","边界条件_平板支座","边界条件_平板铰支座"]].values
t_t_label=srcfst_train[["N"]].values
from sklearn.preprocessing import MinMaxScaler

#t_ftrain_tensor=torch.from_numpy(t_t_value).float()
#t_ltrain_tensor=torch.from_numpy(t_t_label).float()
fit=MinMaxScaler()
t_ftrain_tensor=fit.fit_transform(t_t_value)
t_ftrain_tensor=torch.from_numpy(t_ftrain_tensor).float()
#t_ftrain_tensor=torch.float(t_ftrain_tensor)
t_ltrain_tensor=fit.fit_transform(t_t_label)
t_ltrain_tensor=torch.from_numpy(t_ltrain_tensor).float()

#t_ltrain_tensor=torch.float(t_ltrain_tensor)
#t_ltrain_tensor=torch.from_numpy(t_t_label).float()

v_f=variable(t_ftrain_tensor)
v_l=variable(t_ltrain_tensor)

class srcfstNET(torch.nn.Module):
    def __init__(self,n_f,n_hidden1,n_hidden2,n_hidden3,n_output):#搭建WANGLUO模块
        super(srcfstNET, self).__init__()
        self.hidden1=torch.nn.Linear(n_f,n_hidden1)

        self.hidden2=torch.nn.Linear(n_hidden1,n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.pred=torch.nn.Linear(n_hidden3,n_output)

    def forward(self,v_f):

        v_f=F.sigmoid(self.hidden1(v_f))
        #输出层回归问题一般不同激活函数
        v_f=F.sigmoid(self.hidden2(v_f))
        v_f = F.sigmoid(self.hidden3(v_f))
        v_f=self.pred(v_f)
        return v_f
net=srcfstNET(10,8,5,8,1)
print(net)
#优化

optimizer=torch.optim.Adam(net.parameters(),lr=0.05,betas=(0.7,0.99))
loss_func=torch.nn.MSELoss()
#训练网络
for t in range(2000):
    predic=net(v_f)
    loss=loss_func(predic,v_l)

    print(loss)
    optimizer.zero_grad()#tidu重新置0
    loss.backward()
    optimizer.step()
torch.save(net, 'net_model.pkl')

net_loaded = torch.load('net_model.pkl')


