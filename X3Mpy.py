#%%
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
#%% md
# #### Reading data
#%%
df = pd.read_csv('Challege-Data-V2.tsv', sep='\t')
# df = pd.read_table('Challege-Data-V2.tsv')
# df = pd.read_csv('Challege-Data-V2.tsv', sep='\t',  converters={'waterfall_result': literal_eval})
#%%
df.head(5)
#%%
print(f'df size: {df.size}')
#%% md
# #### Processing device data
#%%
df['device'] = df['device'].apply(lambda x: eval(x))
dfdevice = pd.json_normalize(df['device'])
dfdevice.rename(columns={'type':'device_type', 'model':'device_model', 'w':'device_w', 'h':'device_h', 'memory_total':'device_memory'},
                inplace=True)
df = pd.concat([df.drop(['device'], axis=1), dfdevice], axis=1)
df['device_model'] = df['device_model'].str.extract('(\d+)')
df['device_model'] = df['device_model'].astype(str).astype(int)
#%% md
# #### **Understanding waterfall logic**
#%%
print(df[:1]['waterfall_result'])
print(type(df[:1]['waterfall_result']))
#%%
for i in df[:1]['waterfall_result']:
    for item in eval(i):
        print(item)
        print(type(item))
#%% md
# #### Processing waterfall data, exploding instances
#%%
df['waterfall_result'] = df['waterfall_result'].apply(json.loads)
# df['waterfall_result'] = df['waterfall_result'].apply(lambda x: eval(x))
#%%
for i in df[:1]['waterfall_result']:
    print(type(i))
#%%
df = df.explode('waterfall_result').reset_index(drop=True)
dfwaterfalls = pd.json_normalize(df['waterfall_result'])
dfwaterfalls.rename(columns={'id':'instance_id', 'error':'instance_error'},
                    inplace=True)
df = pd.concat([df.drop(['waterfall_result'], axis=1), dfwaterfalls], axis=1)
#%% md
# #### Adding target: errors are filled instances
#%%
df['instance_error'] = np.where(df['instance_error'].isnull(), 1, 0)
df.rename(columns={'instance_error':'target'}, inplace=True)
#%% md
# #### **Exploring the dataset**
# 
#%%
df.describe()
# Unique: app_id, user_id, waterfall_id
#%%
print(df.event_time.min())
print(df.event_time.max())
print('Normal date ranges')
#%%
df.shape
#%%
print('Checking unique values')
for i in list(df.columns):
    print(f'Distinct values for {i}: {df[i].nunique()}')
#%%
print('Checking nulls')
for i in list(df.columns):
    print(f'Nulls for {i}: {df[i].isnull().sum()}')
#%% md
# #### Checking distribution inside variables
#%%
df.connection_type.value_counts(normalize=True, dropna=False)
#%%
df.device_type.value_counts(normalize=True, dropna=False)
#%%
df.device_model.value_counts(normalize=True, dropna=False)

#%%
df.device_w.value_counts(normalize=True, dropna=False)
#%%
df.device_h.value_counts(normalize=True, dropna=False)
#%% md
# #### Analyzing price
#%%
df['ecpm'].quantile([0.25, 0.5, 0.75])
#%%
sns.violinplot(x=df[df['ecpm']<=150]["ecpm"])
#%%
sns.histplot(df[df['ecpm']<=150]['ecpm'],kde=True,color='purple',bins=30)
#%% md
# #### Price and partners
#%%
sns.violinplot(data=df, x=df[df['ecpm']<=150]["ecpm"], y="partner", height=8.27, aspect=11.7/8.27)
#%%
agg_funcs = {
    'event_id': ['count'],
    'ecpm': ['min', 'mean', 'max'],
    'target': lambda x: (x == 1).sum()
}


partners = df.groupby('partner').agg(agg_funcs).reset_index()
partners.columns = ['partner', 'n_instances', 'min_price', 'mean_price', 'max_price', 'target_1']
partners['fill_rate'] = (partners['target_1']/partners['n_instances']).astype(float)
partners.sort_values(by='fill_rate', ascending=False)
# List of partners with the number of instances, the min/mean/max price, number of filled instances and the fill rate.
# There are 8 partners with a fill rate > 50%.
#%%
print('Partners with >= 0.5 fill rate will be considered as bidders and may be excluded')
bidders = partners[partners['fill_rate']>=0.5].partner.to_list()
bidders
#%%
print('With bidders')
print(df.target.value_counts())
print(df.target.value_counts(normalize=True))
#%%
print('Without bidders')
print(df[~df['partner'].isin(bidders)].target.value_counts())
print(df[~df['partner'].isin(bidders)].target.value_counts(normalize=True))
#%% md
# #### Excluding partners with high fill rate
#%%
print('Excluding bidders')
dfh = df[~df['partner'].isin(bidders)]
#%%
dfh.lifecycle_counter.value_counts()
#%%
dfh.head()
#%% md
# #### Choosing variables
#%%
dfinal = dfh[['connection_type', 'device_type', 'device_model', 'device_w', 'device_h', 'device_memory', 'ecpm', 'target']]
#%%
dfinal.dtypes
#%%
#Categorical data
categorical_cols = ['connection_type', 'device_type']
dfcat = pd.get_dummies(dfinal, columns = categorical_cols, dtype=int)
#%%
dfcat.columns.tolist()
#%%
abs(pd.DataFrame(dfcat.corr().target))
#%% md
# #### Running predictive models
#%%
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#%%
X = dfcat.drop(['target'], axis=1)
y = dfcat['target']
#%%
X.head(3)
#%%
y.head(2)
#%%
y.value_counts(normalize=True)
#%%
print('Oversampling minority class')
oversample = RandomOverSampler(sampling_strategy=0.5)
# undersample = RandomUnderSampler(sampling_strategy=0.5)
# smt = SMOTE()
#%%
X_over, y_over = oversample.fit_resample(X, y)
# X_over, y_over = smt.fit_resample(X, y)
# X_under, y_under = undersample.fit_resample(X, y)
#%%
y_over.value_counts(normalize=True)
#%%
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.3,random_state=101)
#%%
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')
#%%
y_test.value_counts(normalize=True)
#%%
print('Scaling numerical values')
cols = X_train.columns

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train,columns=cols)
X_test = pd.DataFrame(X_test,columns=cols)
#%%
X_train.head(3)
#%%
LR = LogisticRegression()
LRparam_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['lbfgs', 'liblinear', 'saga']
}
LR_search = GridSearchCV(LR, param_grid=LRparam_grid, refit = True, verbose = 3, cv=5)

LR_search.fit(X_train , y_train)
print('Mean Accuracy: %.3f' % LR_search.best_score_)
print('Config: %s' % LR_search.best_params_)
#%%
model = LogisticRegression(
    solver='liblinear',
    C=1,
    max_iter=500)
model.fit(X_train,y_train)
#%%
y_pred = model.predict(X_test)
#%%
cfm = confusion_matrix(y_test, y_pred)
cfm
#%%
print('Accuracy: ', (metrics.accuracy_score(y_test, y_pred)))
print('Recall: ', (metrics.recall_score(y_test, y_pred)))
print('Precision: ', (metrics.precision_score(y_test, y_pred)))
#%%
print('Bad recall, not identifying TPs')
#%%
prediction_prob = model.predict_proba(X_test)[:,1]
prediction_prob[prediction_prob > 0.5] = 1
prediction_prob[prediction_prob <= 0.5] = 0
print(classification_report(y_test,prediction_prob))
#%% md
# #### Testing with K-Fols cross-val
#%%
kf = KFold(n_splits=10, random_state=1, shuffle=True)
#%%
model = LogisticRegression(
    solver='liblinear',
    C=1,
    max_iter=500)
model.fit(X_train, y_train)
#%%
acc = cross_val_score(model, X_over, y_over, cv= kf, scoring="accuracy")
print(f'Accuracy for each fold: {acc}')
print(f'Average accuracy: {"{:.2f}".format(acc.mean())}')
#%%
recall = cross_val_score(model, X, y, cv= kf, scoring='recall')
print('Recall', np.mean(recall), recall)
precision = cross_val_score(model, X, y, cv= kf, scoring='precision')
print('Precision', np.mean(precision), precision)
f1 = cross_val_score(model, X, y, cv= kf, scoring='f1')
print('F1', np.mean(f1), f1)
#%%
print('There are no significant improvements with cross validation')
#%% md
# #### Running NNs
#%%
from tensorflow import keras
from sklearn.metrics import accuracy_score
#%%
X_train.shape[1]
#%%
m = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
m.summary()
#%%
m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#%%
m.fit(X_train, y_train, batch_size = 32, verbose = 2, epochs = 30)
#%%
y_pred_nn= (m.predict(X_train) > 0.5).astype(int)
print('Precision : ', np.round(metrics.precision_score(y_train, y_pred_nn)*100,2))
print('Accuracy : ', np.round(metrics.accuracy_score(y_train, y_pred_nn)*100,2))
print('Recall : ', np.round(metrics.recall_score(y_train, y_pred_nn)*100,2))
print('F1 score : ', np.round(metrics.f1_score(y_train, y_pred_nn)*100,2))
print('AUC : ', np.round(metrics.roc_auc_score(y_train, y_pred_nn)*100,2))
#%%
