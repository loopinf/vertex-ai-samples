# %%
from numpy import dtype
import pandas as pd
import os
import holoviews as hv
import hvplot
import pickle
import gcsfs
import gzip
fs = gcsfs.GCSFileSystem(project='dots-stock')

#%%

full_tech_indi = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210825090905/get-full-tech-indi_-854549233200529408/full_tech_indi_dataset'
df_ = pd.read_csv(full_tech_indi)

#%%
tech_indi_01 = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210825090905/get-tech-indi_-3736852994717646848/df_techini_dataset'
df_tech_01 = pd.read_csv(tech_indi_01)

#%%

adj_price_each = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210825173450/get-adj-prices-01_-8666042786874654720/adj_price_dataset'
# df_adj_price_each = pd.read_csv(adj_price_each)

with open(adj_price_each, 'rb') as f:
    df_adj_price_each = pickle.load(f)
#%%
market_info = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210825083522/get-market-info_3333798420254031872/market_info_dataset'
# df_market_info = pd.read_csv(market_info,
#                             index_col=0,
#                             dtype={'날짜':str, '종목코드':str}
#                             ).reset_index(drop=True)

df_market = pd.read_pickle(market_info)



# %%
feat_path = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210825173450/get-features_-1748513759233572864/features_dataset'
df = pd.read_csv(feat_path)
# with open(feat_path, 'rb') as f:
#     df_feat = pickle.load(f)
# %%
