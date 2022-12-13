# @Time : 2022/5/23 16:12
# @Author : hongzt
# @File : srcfst_lgbm
import pandas as pd

import numpy as np
import warnings
warnings.filterwarnings("ignore")
trains= pd.read_excel('F:\\database\\srcfst\\srcfstmax\\ssrcfst1.xlsx')
trainm=pd.read_excel('F:\\database\\srcfst\\srcfstmax\\lsrcfst.xlsx')
rmse=[]
rmse1=[]
rmse2=[]
dems=trains[["D","T","L","fty","As","fsy","fc"]].values
objects=trains[["N"]].values
demm=trainm[["D","T","L","fty","As","fsy","fc"]].values
objectm=trainm[["N"]].values

from sklearn.model_selection import train_test_split
x_trains,x_tests,y_trains,y_tests=train_test_split(dems,objects,test_size=0.3,shuffle=True)
x_trainm,x_testm,y_trainm,y_testm=train_test_split(demm,objectm,test_size=0.3,shuffle=True)
import joblib
from sklearn.model_selection import KFold


from sklearn.metrics import mean_squared_error
import lightgbm as lgbm
from sklearn.model_selection import KFold


kf = KFold(n_splits=5)
for train_indices, test_indices in kf.split(dems):
    X_train, X_test = dems[train_indices], dems[test_indices]
    Y_train, Y_test = objects[train_indices], objects[test_indices]
    lgbmRLRs = lgbm.LGBMRegressor(colsample_bytree=0.6, subsample=0.5, max_depth=3, n_estimators=100, max_delta_step=0,
                                 learning_rate=0.32, reg_alpha=0.5, min_child_weight=1.99, reg_lambda=0.229)

    lgbmRLRs.fit(X_train, Y_train)

    y_pred = lgbmRLRs.predict(X_test)
    RSM = np.sqrt(mean_squared_error(Y_test, (y_pred)))
    rmse.append(RSM)

    if RSM <= min(rmse) and len(rmse) != 0:
        joblib.dump(filename='lgbm_srcfsts.model1', value=lgbmRLRs)

    rmse_1 = np.sqrt(mean_squared_error(y_trains, lgbmRLRs.predict(x_trains)))

    rmse1.append(rmse_1)
print(rmse)
print(np.mean(rmse), np.mean(rmse1))


RMSE2 = []
RMSE3=[]

for train_indicem, test_indicem in kf.split(demm):
    X_trainM, X_testM = demm[train_indicem], demm[test_indicem]
    Y_trainM, Y_testM = objectm[train_indicem], objectm[test_indicem]
    lgbmRLRm1 = lgbm.LGBMRegressor()
    lgbmRLRm1.fit(X_trainM, Y_trainM)

    y_predM = lgbmRLRm1.predict(X_testM)
    RSMm = np.sqrt(mean_squared_error(Y_testM, (y_predM)))

    RMSE2.append(RSMm)

    if RSMm <= min(RMSE2) and len(RMSE2) != 0:
        joblib.dump(filename='lgbm_m_scrcfst.model1', value=lgbmRLRm1)

print(RMSE2)

