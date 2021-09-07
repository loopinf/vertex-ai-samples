#%%

# imports
from typing import Protocol
import pandas as pd
import FinanceDataReader as fdr
import pickle5 as pickle



#%%
# loading dfs

# 이전 컴포넌트에서 나온 예측 결과는 기존 예측 결과와 이미 합쳐저 있어야 함
# 이 컴포넌트에서는 합쳐져 있는 결과를 불러오는 것을 가정하였음

# path_df_pred_result = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210901225806/create-model-and-prediction-02_5031796217261064192/prediction_result_02'
# path_df_pred_result = 'gs://pipeline-dots-stock/prediction_results_3d_close_10_v01/df_predic_result_3d_close_10_v01_price_updated.pkl'
df_pred_result = pd.read_pickle('result_bong04_06_rp02.pkl')
# df_pred_result.rename(columns={'날짜':'date', '종목코드':'code'}, inplace=True)

# %%

# check date

l_dates = df_pred_result.date.unique().tolist()

dates_to_update = l_dates[:-3]

# df_to_hold = df_pred_result[~df_pred_result.date.isin(dates_to_update)]
df_to_update = df_pred_result[df_pred_result.date.isin(dates_to_update)]

codes_to_update = df_to_update.code.unique().tolist()

#%%

def get_price_adj(code, start):
      return fdr.DataReader(code, start=start)    
    #---------------------------------------------------------------------------
def get_price(codes, date_start):

    df_price = pd.DataFrame()

    for code in codes :
    
        df_ = get_price_adj(code, date_start)
        df_['code'] = code
        df_price = df_price.append(df_)

        # print(df_price.shape, code)
    
    return df_price
# %%
date_start = dates_to_update[0]
df_price = get_price(codes_to_update, date_start)

df_price.reset_index(inplace=True)
df_price.columns = df_price.columns.str.lower()
df_price['date'] = df_price.date.dt.strftime('%Y%m%d')

# %%

def get_price_tracked(df):

    df_ = df.copy()
    df_.sort_values(by='date', inplace=True)
    df_['c_1'] = df_.close.shift(-1)
    df_['c_2'] = df_.close.shift(-2)
    df_['c_3'] = df_.close.shift(-3)

    return df_

#%%

df_price_updated  = df_price.groupby('code').apply(lambda df: get_price_tracked(df))
df_price_updated = df_price_updated.reset_index(drop=True)

# %%

try :
    df_to_update.drop(columns=['c_1', 'c_2', 'c_3'], inplace=True)
except :
    pass

df_to_update = df_to_update.merge(
                        df_price_updated,
                        left_on=['date', 'code'],
                        right_on=['date', 'code'] )
df_to_update.fillna(0, inplace=True)



# with open('genesis08_result.pkl', 'wb') as f:
#     pickle.dump(df_to_update, f)

#%%

# df_to_update.to_pickle('genesis08_result.pkl', protocol=1)

# with open('bong01.pkl', 'wb') as f:
#     pickle.dump(df_to_update, f)
# %%

df_x = df_to_update[['name', 'code', 'date', 'close', 'c_1', 'c_2', 'c_3']]
df_y = df_to_update[['name', 'code', 'date']]

df_y['c1_r'] = df_to_update['c_1']/df_to_update['close']
df_y['c2_r'] = df_to_update['c_2']/df_to_update['close']
df_y['c3_r'] = df_to_update['c_3']/df_to_update['close']
# %%
l_dates = df_y.date.unique().tolist()

return_mean = []
for date in l_dates :
    df_ = df_y[df_y.date == date].head(10)
    # print(date, df_.shape)

    mean_ = df_['c3_r'].mean()
    print(mean_)
    return_mean.append(mean_)

df__ = pd.DataFrame(list(zip(l_dates, return_mean)),
                        columns=['date', 'mean'])

a = 0

for r in return_mean:
    a = a + (r-1)*100

print('sum', a)
#%%
df__.to_excel('output04_06_rp02.xlsx')
print('done!')
# %%
