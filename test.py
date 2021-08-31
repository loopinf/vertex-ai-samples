#%%
import pandas as pd

#%%

path = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210831132626/get-ml-dataset_-3017684429221920768/ml_dataset'


# %%
df = pd.read_pickle(path)
# %%
