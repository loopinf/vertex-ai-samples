#%%
import os
import pandas as pd
import numpy as np

folder = '/Users/DSK/Downloads/'
file_bro = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210821174300_get-bros_1493488634240696320_bros_univ_dataset'
file_market = 'pipeline_root_shkim01_516181956427_ml-with-all-items-20210821175544_get-market-info_214466340067475456_market_info_dataset'
path = os.path.join(folder, file_market)
path2 = os.path.join(folder, file_bro)

df_market = pd.read_csv(path,
                        index_col=0,
                        dtype={'날짜':str, '등락률': 'double'}
                        ).reset_index(drop=True)

dates_in_set = df_market.날짜.unique().tolist()
dates_on_train = df_market.날짜.unique().tolist()[-20:]

# 등락률 -1 
df_market = df_market.sort_values('날짜')
df_market['return_-1'] = df_market.groupby('종목코드').등락률.shift(1)

#df_ed 가져오기
df_ed = pd.read_csv(path2, index_col=0).reset_index(drop=True)
df_ed_r = df_ed.copy() 
df_ed_r.rename(columns={'target':'source', 'source':'target'}, inplace=True)
df_ed2 = df_ed.append(df_ed_r, ignore_index=True)
df_ed2['date'] = pd.to_datetime(df_ed2.date).dt.strftime('%Y%m%d')

cols = ['종목코드', '날짜']
df_mkt_ = df_market[cols]

cols_market = ['종목코드','날짜','등락률','return_-1']
cols_bro = ['source','target','period','date']

# merge
df_ed2_1 = ( df_ed2[cols_bro]
                .merge(df_market[cols_market], 
                    left_on=['target','date'],
                    right_on=['종목코드','날짜'])
                .rename(columns={'등락률':'target_return',
                'return_-1':'target_return_-1'}))
df_ed2_1 = df_ed2_1[['source', 'target', 'period', 'date', 
                    'target_return', 'target_return_-1']]
#%%
df_tmp = df_mkt_.merge(df_ed2_1, 
        left_on=['날짜','종목코드'], 
        right_on=['date', 'source'], 
        suffixes=('','_x'),
        how='left')
df_tmp.drop(columns=['종목코드','날짜'], inplace=True)
df_tmp.dropna(subset=['target'], inplace=True)


#%%
def get_upbro_ratio(df):
    '''df : '''
    return (
            sum(df.target_return > 0) /
            df.shape[0], # 그날 상승한 친구들의 비율
            df.shape[0], # 그날 친구들 수
            df.target_return.mean(), # 그날 모든 친구들 상승률의 평균
            df[df.target_return > 0].target_return.mean() # 그날 오른 친구들의 평균
            # 전날 친구들 평균상승률
            # 전날 상승한 친구들 비율
            # 전달 오른 친구들 평균

            )

bro_up_ratio = (df_tmp.groupby(['date','source','period'])
    .apply(lambda df: get_upbro_ratio(df))
    .reset_index()
    .rename(columns={0:'bro_up_ratio'})
    )
#%%
bro_up_ratio[['bro_up_ratio','n_bros']] = \
    pd.DataFrame(bro_up_ratio.bro_up_ratio.tolist(), index=bro_up_ratio.index) 
#%%
df_tmp = df_tmp.merge(bro_up_ratio, on=['date','source','period'], how='left')
df_tmp['up_bro_ratio_20'] = df_tmp[df_tmp.period == 20].bro_up_ratio
df_tmp['up_bro_ratio_40'] = df_tmp[df_tmp.period == 40].bro_up_ratio
df_tmp['up_bro_ratio_60'] = df_tmp[df_tmp.period == 60].bro_up_ratio
df_tmp['up_bro_ratio_90'] = df_tmp[df_tmp.period == 90].bro_up_ratio
df_tmp['up_bro_ratio_120'] = df_tmp[df_tmp.period == 120].bro_up_ratio

df_tmp.fillna(0, inplace=True) #친구가 없는 종목의 bro_up_ratio를 0으로 만들기

df_tmp['n_bro_20'] = df_tmp[df_tmp.period == 20].n_bros
df_tmp['n_bro_40'] = df_tmp[df_tmp.period == 40].n_bros
df_tmp['n_bro_60'] = df_tmp[df_tmp.period == 60].n_bros
df_tmp['n_bro_90'] = df_tmp[df_tmp.period == 90].n_bros
df_tmp['n_bro_120'] = df_tmp[df_tmp.period == 120].n_bros

df_tmp.fillna(0, inplace=True) #친구가 없는 종목의 n_bros를 0으로 만들기

df_tmp.drop(columns=['bro_up_ratio','n_bros'], inplace=True)


# %%
df_tmp.groupby('')

# %%
