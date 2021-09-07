#%%

import pandas as pd
# import pickle
import numpy as np
from collections import Counter

import FinanceDataReader as fdr
# import pickle

# %%
folder = '/Users/seunghankim/Downloads/'
feat = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210823113318_get-features_-363683262096211968_features_dataset'
target = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210823002313_get-target_-1738969998304477184_df_target_dataset'
tech = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210823003125_get-full-tech-indi_217281089834582016_full_tech_indi_dataset'


path = 'gs://pipeline-dots-stock/bong_price_updated/bong_04.pkl'
#%%
df__ = pd.read_pickle(path) 

df_ = df__.copy()

# with open(path, 'rb') as f:
  # df_ = pickle.load(f)

#%%

# df_ = df_.reset_index(drop=True)

df_ = df_[['name', 'code', 'date', 'Prediction', 'Proba01', 'Proba02',
            'high', 'low', 'volume', 'change', 'c_1', 'c_2', 'c_3', 'close'
            ]]


l_dates = df_.date.unique().tolist()

l_dates_to_update = l_dates #[-5:]

df_to_hold = df_[~df_.date.isin(l_dates_to_update)]
df_to_update = df_[df_.date.isin(l_dates_to_update)]

#%%

codes_to_update = df_to_update.code.unique().tolist()

def get_price_adj(code, start):
  return fdr.DataReader(code, start=start)    

def get_price(codes, date_start):

    df_price = pd.DataFrame()
    for code in codes :      
        df_ = get_price_adj(code, date_start)
        df_['code'] = code
        df_price = df_price.append(df_)
 
    return df_price

date_start = l_dates_to_update[0]
df_price = get_price(codes_to_update, date_start)
df_price.reset_index(inplace=True)
df_price.columns = df_price.columns.str.lower()
df_price['date'] = df_price.date.dt.strftime('%Y%m%d')


def get_price_tracked(df):

  df_ = df.copy()
  df_.sort_values(by='date', inplace=True)
  df_['c_1'] = df_.close.shift(-1)
  df_['c_2'] = df_.close.shift(-2)
  df_['c_3'] = df_.close.shift(-3)

  return df_

df_price_updated  = df_price.groupby(['code']).apply(lambda df: get_price_tracked(df))
df_price_updated = df_price_updated[['date','c_1', 'c_2', 'c_3', 'close']]
df_price_updated = df_price_updated.reset_index()
df_price_updated.drop(columns=['level_1'], inplace=True)


#%%
df_to_update.drop(columns=['c_1', 'c_2', 'c_3', 'close'], inplace=True)

#%%
df_to_update = df_to_update.merge(
                      df_price_updated,
                      left_on=['date', 'code'],
                      right_on=['date', 'code'],
                      how='inner' )

df_to_update.fillna(0, inplace=True)


#%%

df_updated = pd.DataFrame()
# df_to_update.index = df_to_update.index +1000
# df_updated = pd.concat([df_to_hold, df_to_update])
df_updated = df_updated.append(df_to_hold)

# df_updated = df_to_hold.join(df_to_update, how='outer')
# df_updated = df_updated.reset_index(drop=True)
# df_updated.to_pickle(path)

# %%
df_updated = df_updated.append(df_to_update)
# %%
