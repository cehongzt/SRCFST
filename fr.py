# @Time : 2022/4/7 9:07
# @Author : hongzt
# @File : random f
import pandas as  pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
#导入数据srcfst注意：此处不能使用汉语的路径
srcfst_train=pd.read_csv("F:\\database\\srcfst\\srcfstmax\\train.csv")
srcfst_test=pd.read_csv("F:\\database\\srcfst\\srcfstmax\\test.csv")

srcfst_train=pd.get_dummies(srcfst_train)#字符串独热编码
#调用值


t_t_value=srcfst_train[["D","T","L","长细比","fty","As","fsy","fc","边界条件_平板支座","边界条件_平板铰支座"]].values
t_t_label=srcfst_train[["N"]].values

#from sklearn import preprocessing
#t_t_value=preprocessing.StandardScaler().fit_transform(t_t_value)

#归一化
from sklearn import preprocessing
#t_t_value=preprocessing.StandardScaler().fit_transform(t_t_value)
#划分测试数据集
from sklearn.model_selection import train_test_split
x_train,x_test,y_trian,y_test=train_test_split(t_t_value,t_t_label,test_size=0.3,shuffle=True)

#导入模型
from sklearn.ensemble import RandomForestRegressor
srcfr=RandomForestRegressor()
import  joblib
srcfr.fit(x_train,y_trian)
y_pred=srcfr.predict(x_test)

from sklearn.metrics import mean_squared_error
#print(f'root mean square eoore:{np.sqrt(mean_squared_error(y_test,(y_pred)))}')
#y_pred=srcfr.predict([[300,3.87,900,12,311,2892,311,29.3,1,0]])
#print(y_pred)
from sklearn.metrics import mean_squared_error
#交叉验证
from sklearn.model_selection import KFold
kf=KFold(n_splits=5)
rmse=[]
for train_indices,test_indices in kf.split(t_t_value):
    X_train, X_test = t_t_value[train_indices], t_t_value[test_indices]
    Y_train, Y_test = t_t_label[train_indices], t_t_label[test_indices]
    fr=RandomForestRegressor()
    fr.fit(X_train,Y_train)
    y_pred = fr.predict(X_test)

    RSM=np.sqrt(mean_squared_error(Y_test,abs(y_pred)))
    rmse.append(RSM)
    if RSM <= min(rmse) and len(rmse) !=0 :
        joblib.dump(filename='fr.model1', value=fr)



print(rmse)
predietmodel= joblib.load('fr.model1')
y_pred_train=predietmodel.predict(x_train)

RSM1=np.sqrt(mean_squared_error(y_trian,abs(y_pred_train)))

print(f'TRAIN_rmse:,{(RSM1)}')
print(f'average rmse:{np.mean(rmse)}')
from sklearn.metrics import mean_absolute_error
RSMmabe2=np.sqrt(mean_absolute_error(y_trian,abs(y_pred_train)))
print(RSMmabe2)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
print('mape',mape(y_trian,abs(y_pred_train)))

from sklearn.metrics import r2_score
RSM2=np.sqrt(r2_score(y_trian,abs(y_pred_train)))
print(RSM2)

