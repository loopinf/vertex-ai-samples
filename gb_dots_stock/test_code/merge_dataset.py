#%%

import pandas as pd
# import pickle
import numpy as np
from collections import Counter

# %%
folder = '/Users/seunghankim/Downloads/'
feat = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210823113318_get-features_-363683262096211968_features_dataset'
target = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210823002313_get-target_-1738969998304477184_df_target_dataset'
tech = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210823003125_get-full-tech-indi_217281089834582016_full_tech_indi_dataset'

# 등락률 -1 
df_market = df_market.sort_values('날짜')
df_market['return_-1'] = df_market.groupby('종목코드').등락률.shift(1)

#df_ed 가져오기
path = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210825210134/get-bros_9060125346455617536/bros_univ_dataset'
df_ed = pd.read_pickle(path)

df_ed_r = df_ed.copy() 
df_ed_r.rename(columns={'target':'source', 'source':'target'}, inplace=True)
df_ed2 = df_ed.append(df_ed_r, ignore_index=True)
df_ed2['date'] = pd.to_datetime(df_ed2.date).dt.strftime('%Y%m%d')

cols = ['종목코드', '종목명', '날짜', '순위_상승률']
df_mkt_ = df_market[cols]

cols_market = [ '종목코드','날짜','등락률','return_-1']
cols_bro = ['source','target','period','date']

# merge
df_ed2_1 = (df_ed2[cols_bro]
                .merge(df_market[cols_market], 
                    left_on=['target','date'],
                    right_on=['종목코드','날짜'])
                .rename(columns={'등락률':'target_return',
                'return_-1':'target_return_-1'}))
df_ed2_1 = df_ed2_1[['source', 'target', 'period', 'date', 
                    'target_return', 'target_return_-1']]

df_tmp = df_mkt_.merge(df_ed2_1, 
        left_on=['날짜','종목코드'], 
        right_on=['date', 'source'], 
        suffixes=('','_x'),
        how='left')
df_tmp.drop(columns=['종목코드','날짜'], inplace=True)
df_tmp.dropna(subset=['target'], inplace=True)

def get_upbro_ratio(df):

    return (
            sum(df.target_return > 0) /
            df.shape[0], # 그날 상승한 친구들의 비율
            df.shape[0], # 그날 친구들 수
            df.target_return.mean(), # 그날 모든 친구들 상승률의 평균
            df[df.target_return > 0].target_return.mean(), # 그날 오른 친구들의 평균
            df['target_return_-1'].mean(),# 전날 친구들 평균상승률
            sum(df['target_return_-1'] > 0) / df.shape[0],# 전날 상승한 친구들 비율
            df[df['target_return_-1'] > 0]['target_return_-1'].mean(),# 전날 상승한 친구들 평균

            )

bro_up_ratio = (df_tmp.groupby(['date','source','period'])
    .apply(lambda df: get_upbro_ratio(df))
    .reset_index()
    .rename(columns={0:'bro_up_ratio'})
    )
#%%
#5
bro_up_ratio[['bro_up_ratio','n_bros', 'all_bro_rtrn_mean', 'up_bro_rtrn_mean',
                'all_bro_rtrn_mean_ystd', 'bro_up_ratio_ystd', 'up_bro_rtrn_mean_ystd']] = \
    pd.DataFrame(bro_up_ratio.bro_up_ratio.tolist(), index=bro_up_ratio.index) 



df_rank = df_mkt_.copy()

df_rank['in_top30'] = df_rank.순위_상승률 < 30
df_rank['rank_mean_10'] = df_rank.groupby('종목코드')['순위_상승률'].transform(
                            lambda x : x.rolling(10, min_periods=1).mean()
                        )

df_rank['rank_mean_5'] = df_rank.groupby('종목코드')['순위_상승률'].transform(
                            lambda x : x.rolling(5, min_periods=1).mean()
                        )

df_rank['in_top_30_5'] = df_rank.groupby('종목코드')['in_top30'].transform(
                            lambda x : x.rolling(5, min_periods=1).sum()
                        )

df_ml_dataset = (df_feats.merge(df_target,
                            left_on=['source', 'date'],
                            right_on=['code', 'date'],
                            how='left'))

df_ml_dataset = (df_ml_dataset.merge(df_tech,
                            left_on=['source', 'date'],
                            right_on=['code', 'date'],
                            how='left'))


df_ml_dataset.dropna(inplace=True)

# df_feats.fillna(0, inplace=True)
df_feats.drop(columns=['source', 'date'], inplace=True)
df_feats.rename(columns={'종목코드':'code', '종목명':'name', '순위_상승률':'rank', '날짜':'date'}, inplace=True)
df_feats.fillna(0, inplace=True)

# df_feats.to_pickle(features_dataset.path)
# %%

# %%
path = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/ml-with-all-items-20210825225701/get-features_7116259157291827200/features_dataset'

df_ = pd.read_pickle(path)
# %%
