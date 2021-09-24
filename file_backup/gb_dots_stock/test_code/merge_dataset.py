#%%

import pandas as pd
import numpy as np
from collections import Counter

import sys
import FinanceDataReader as fdr
from catboost import CatBoostClassifier

import hvplot.pandas
import holoviews as hv


# %%
results = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/model-backtesting-m19-5-01-20210924214724/add-prices-n-returns_5020783508797259776/price_n_return_updated_dataset'

df = pd.read_pickle(results)

#%%
hv.extension('bokeh')

df.hvplot(x='Proba02', y='r3', kind='scatter')


#%%
df['ratio%'].mean()
#%%

cols_indicator = ['code', 'name','date']
X_pred = df_predict.drop(columns=cols_indicator)
X_indi = df_predict[cols_indicator]

def get_pred_single(model):
    model_ = CatBoostClassifier()
    model_.load_model(model)
    pred = model_.predict(X_pred)
    pred = pd.DataFrame(pred, columns=['Prediction']).reset_index(drop=True)
    pred_prob = model_.predict_proba(X_pred)
    pred_prob = pd.DataFrame(pred_prob, columns=['Prob01','Prob02']).reset_index(drop=True)
    return pd.concat([X_indi, pred, pred_prob], axis=1)

#%%
df_pred01 = get_pred_single(model_01)
df_pred02 = get_pred_single(model_02)
df_pred03 = get_pred_single(model_03)

df_pred_all = pd.concat([df_pred01, df_pred02, df_pred03], axis=0)

df_pred_mean= df_pred_all.groupby(['name', 'code', 'date']).mean() # apply mean to duplicated recommends
df_pred_mean = df_pred_mean.reset_index()
df_pred_mean = df_pred_mean.sort_values(by='Prob02', ascending=False) # high probability first

df_pred_mean.drop_duplicates(subset=['code', 'date'], inplace=True) 

# Load stored prediction result
try :
    df_pred_stored = pd.read_pickle(f'/gcs/pipeline-dots-stock/bong_predictions/bong_{ver}.pkl')
except :
    df_pred_stored = pd.DataFrame()

dates_in_stored = df_pred_stored.date.unique().tolist()

if df_pred_mean.date.iloc[0] not in dates_in_stored:
    df_pred_new = df_pred_stored.append(df_pred_mean)
else :
    df_pred_new = df_pred_stored

# df_pred_new.to_pickle(daily_recom_dataset.path)
# df_pred_new.to_pickle(f'/gcs/pipeline-dots-stock/bong_predictions/bong_{ver}.pkl')

# %%
날짜