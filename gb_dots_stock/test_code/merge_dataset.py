#%%
import pandas as pd

#%%
path_market = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210826072355/get-market-info_-5428517604748689408/market_info_dataset'
path_ml_data = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210826072355/get-ml-dataset_8406540450533474304/ml_dataset'
path_tech = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210826072355/get-full-tech-indi_3794854432106086400/full_tech_indi_dataset'
path_target = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210826072355/get-target_-816831586321301504/df_target_dataset'


df_market = pd.read_pickle(path_market)
df_ml = pd.read_pickle(path_ml_data)
df_tech = pd.read_pickle(path_tech)
df_target = pd.read_pickle(path_target)

# %%
date = '20210824'
df_market_ = df_market[df_market.날짜 == date]
df_ml_ = df_ml[df_ml.date == date]
# %%
l_codes_market = df_market_.종목코드.to_list()
l_codes_ml_data = df_ml_.code.to_list()

l_diff = set(l_codes_market) - set(l_codes_ml_data)
# %%
