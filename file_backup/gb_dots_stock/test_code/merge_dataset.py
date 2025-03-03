#%%

import pandas as pd
import numpy as np
from collections import Counter

import sys
import FinanceDataReader as fdr
from catboost import CatBoostClassifier

import hvplot.pandas
import holoviews as hv


# %%
results = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/gb-pipeline-calc-corr-rolling-5-120days-20211111151359/calc-corr-rolling5_1411742143738806272/corr_rolling5_dataset'

df = pd.read_pickle(results)


#%%
print('m19-11-05')
l_dates = df.date.unique().tolist()

for i in [5, 10, 15, 20, 25, 30]:

    l_return = []

    for date in l_dates :
        df_ = df[df.date == date].head(i)
        r_mean = df_.r1.mean()
        l_return.append(r_mean)

    print(i, sum(l_return))
     
# %%

def get_final_return(df):
    df_ = df.head(i)
    df['fr'] = df_.r1.mean()
    return df

i = 10

df__ = df.groupby('date').apply(lambda df : get_final_return(df))
df__ = df__.drop_duplicates(subset=['date'])
# %%
hv.extension('bokeh')
df__.hvplot(x='date', y='fr')
# %%
