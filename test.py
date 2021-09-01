#%%
import pandas as pd

#%%

path = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210831132626/create-model-and-prediction-01_441080084598620160/prediction_result_01'


# %%
df = pd.read_pickle(path)
# %%
