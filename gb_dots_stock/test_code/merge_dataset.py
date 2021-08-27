#%%

import pandas as pd
# import pickle
import numpy as np
from collections import Counter

# %%
folder = '/Users/seunghankim/Downloads/'
feat = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210823113318_get-features_-363683262096211968_features_dataset'
target = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210823002313_get-target_-1738969998304477184_df_target_dataset'
tech = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210823003125_get-full-tech-indi_217281089834582016_full_tech_indi_dataset'

#%%
df_feats = pd.read_pickle(features_dataset.path)  

  df_target = pd.read_pickle(target_dataset.path)
  df_target['date'] = pd.to_datetime(df_target.date).dt.strftime('%Y%m%d')

  df_tech = pd.read_pickle(tech_indi_dataset.path)
  df_tech['date'] = pd.to_datetime(df_tech.date).dt.strftime('%Y%m%d')

  df_ml_dataset = (df_feats.merge(df_target,
                            left_on=['code', 'date'],
                            right_on=['code', 'date'],
                            how='left'))

  df_ml_dataset = (df_ml_dataset.merge(df_tech,
                              left_on=['code', 'date'],
                              right_on=['code', 'date'],
                              how='left'))

  df_ml_dataset.dropna(inplace=True)

  df_ml_dataset.to_pickle(ml_dataset.path)