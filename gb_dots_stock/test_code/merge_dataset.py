# %%
from numpy import dtype
import pandas as pd
import os

# %%
folder = '/Users/seunghankim/Downloads/'
feat = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210825083522_get-features_-8195416625814437888_features_dataset'
target = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210825083522_get-target_7945484438681419776_df_target_dataset'
tech = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210825090905_get-full-tech-indi_-854549233200529408_full_tech_indi_dataset'

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

#%%

df_ml_dataset = (df_feats.merge(df_target,
                            left_on=['code', 'date'],
                            right_on=['code', 'date'],
                            how='left'))

#%%
df_ml_dataset = (df_ml_dataset.merge(df_tech,
                            left_on=['code', 'date'],
                            right_on=['code', 'date'],
                            how='left'))

#%%
# df_ml_dataset.dropna(inplace=True)


# %%
df1 = pd.DataFrame({'a': ['foo', 'bar'], 'b': [1, 2]})
df2 = pd.DataFrame({'a': ['foo', 'baz'], 'c': [3, 4]})

df1.merge(df2, how='left', on='a')

# %%
