
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
get_ipython().magic(u'matplotlib inline')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, KFold
from sklearn import metrics
import xgboost as xgb
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, MatrixFactorization, MICE
from sklearn import preprocessing
import operator



def transform_data(data):
    #data.loc[data['Var1']>=300,'Var1'] = data[data['Var1']>=300]['Var1'].median()
    data['Date'] = pd.to_datetime(data['Date'], dayfirst = True )
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    
    data = data.drop(labels=[ 'Date' , 'ID',], axis=1 )
    columns = data.columns.values
    
    for col in columns:
        data[col].fillna(data[col].mean(), inplace=True)
        if (col not in['Park_ID', 'Date','Location_Type', 'Season','Month', 'Day', 'WeekDay','WeekOfYear', 'Is789Day', 'Is78Month']):
            data[col] = StandardScaler().fit_transform(data[col].reshape(-1,1))
    
    return data;

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
train = train[train['Park_ID'] != 19]

df = train
df= df.drop(labels='Footfall', axis=1)
df= df.append(test, ignore_index=True)
df = transform_data(df) 


df_train_data = df.ix[0:train.shape[0]-1,]
df_train_labels = train['Footfall']
df_test_data = df.ix[train.shape[0]:df.shape[0],]

print (df_train_data.shape, df_test_data.shape, df_train_labels.shape)






# In[61]:

def carryForwardImputation(trainData):
    impTrainData = trainData.copy()
    isNullPresentInCol = [ key for key, val in trainData.isnull().sum().iteritems() if val != 0]
    # print isNullPresentInCol
    # print '>>>>', impTrainData['Direction_Of_Wind'][840]

    for colName in isNullPresentInCol:
        nullValIdxs = trainData[colName].index[trainData[colName].apply(np.isnan)]
        length = len(nullValIdxs)
        # print nullValIdxs
        #imputing the first missing value.
        i = 0
        while i<length:
            startInd = i
            while (i+1)<length and nullValIdxs[i]+1 == nullValIdxs[i+1]:
                i += 1
            if nullValIdxs[startInd]!=0 and nullValIdxs[i] != len(trainData.index)-1:
                firstVal = trainData[colName][nullValIdxs[startInd]-1]
                lastVal = trainData[colName][nullValIdxs[i]+1]
                incVal = (lastVal - firstVal)/(i-startInd+1)
                for j in xrange(i-startInd+1):
                    impTrainData[colName][nullValIdxs[startInd+j]] = firstVal + (j+1)*incVal
                    
            i += 1
    # print '<<<', impTrainData['Direction_Of_Wind'][840]
    return impTrainData



def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)

def rmspe(y, yhat):
    return np.sqrt(np.mean(((y - yhat)/y) ** 2))



params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.1,
          "max_depth": 10,
          "subsample": 0.85,
          "colsample_bytree": 0.4,
          "min_child_weight": 10,
          "silent": 1,
          "thread": 1,
          "seed": 1301
          }
num_boost_round = 1200

X_train, X_test, y_train, y_test = train_test_split( df_train_data, df_train_labels, test_size=0.3, random_state=42) 
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)



dtrain = xgb.DMatrix(X_train, y_train_log)
dvalid = xgb.DMatrix(X_test, y_test_log)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=30, feval = rmspe_xg, verbose_eval=True)

print("Validating")
predicted_labels = gbm.predict(xgb.DMatrix(X_test))
#error = np.sqrt(metrics.mean_squared_error(y_test, predicted_labels))
error = rmspe(y_test, np.expm1(predicted_labels))
print('RMSE: {:.6f}'.format(error))


# In[54]:

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()


# In[68]:

ceate_feature_map(df_train_data.columns.values)
importance = gbm.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(8, 20))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')



# In[69]:

predictions = pd.DataFrame(np.expm1(gbm.predict(xgb.DMatrix(df_test_data))))
results = pd.DataFrame();
results['ID'] = test['ID']
results['Footfall'] = np.round(predictions)
timestr = time.strftime("%Y%m%d-%H%M%S")
results.to_csv('submission' + timestr + ".csv", index=False)

