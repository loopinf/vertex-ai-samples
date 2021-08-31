#%%
from re import A
from typing import Awaitable, AwaitableGenerator
import pandas as pd

#%%
path_feat = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210826072355/get-features_6100697441319780352/features_dataset'
path_ml_data = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210831074629/get-ml-dataset_891440047335669760/ml_dataset'
# path_tech = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210826072355/get-full-tech-indi_3794854432106086400/full_tech_indi_dataset'
# path_target = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210826072355/get-target_-816831586321301504/df_target_dataset'

path_tmp = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210831114324/create-model-and-prediction-01_-6413398548259274752/prediction_result_01'

df_feat = pd.read_pickle(path_feat)
df_ml = pd.read_pickle(path_ml_data)
# df_tech = pd.read_pickle(path_tech)
# df_target = pd.read_pickle(path_target)

df = pd.read_pickle(path_tmp)

# df = df[['종목명', '날짜', 'Proba02']]

# %%
date = '20210824'
df_market_ = df_market[df_market.날짜 == date]
df_ml_ = df_ml[df_ml.date == date]
# %%
l_codes_market = df_market_.종목코드.to_list()
l_codes_ml_data = df_ml_.code.to_list()

l_diff = set(l_codes_market) - set(l_codes_ml_data)
# %%