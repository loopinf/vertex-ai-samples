#%%

# imports
import pandas as pd

#%%

today = '20210831'

#%%

# load prediction results
pred_result_path = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210901105954/create-model-and-prediction-01_-2817977933245710336/prediction_result_01'
df_pred_results = pd.read_pickle(pred_result_path)

# %%

# get dates

l_dates = df_pred_results.date.unique().tolist()

last_date = l_dates[-1]

if last_date < today :
    dates_updating = l_dates[]



# %%

# 1. get prediction result (df)
# select dates
# 2. get codes from the df
# 3. get price from fdr
# 4. shift
# 5. merge by code and date