# %%
from numpy import dtype
import pandas as pd
import os

# %%
folder = '/Users/seunghankim/Downloads/'
feat = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210822201701_get-features_-6999174363073216512_features_dataset'
target = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210823002313_get-target_-1738969998304477184_df_target_dataset'
tech = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210823003125_get-full-tech-indi_217281089834582016_full_tech_indi_dataset'

path_feat = os.path.join(folder, feat)
path_target = os.path.join(folder, target)
path_tech = os.path.join(folder, tech)

#%%
df_feats = pd.read_csv(path_feat,
                        index_col=0,
                        dtype={'date':str},
                              ).reset_index(drop=True)
df_target = pd.read_csv(path_target,
                        index_col=0,
                        dtype={'code':str},
                            ).reset_index(drop=True)
df_target['date'] = pd.to_datetime(df_target.date).dt.strftime('%Y%m%d')


df_tech = pd.read_csv(path_tech,
                        index_col=0,
                            ).reset_index(drop=True)
df_tech['date'] = pd.to_datetime(df_tech.date).dt.strftime('%Y%m%d')

df_ml_dataset = (df_feats.merge(df_target,
                            left_on=['source', 'date'],
                            right_on=['code', 'date']))

df_ml_dataset = (df_ml_dataset.merge(df_tech,
                            left_on=['source', 'date'],
                            right_on=['code', 'date']))
# %%
