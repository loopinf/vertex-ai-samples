# %%

import pandas as pd
import os

import numpy as np

from collections import Counter
import functools
# from trading_calendars import get_calendar

# %%
folder='/Users/seunghankim/Downloads'
file='pipeline_root_shkim01_516181956427_ml-with-all-items-20210821225639_get-market-info_-2727510116512301056_market_info_dataset'
file2='pipeline_root_shkim01_516181956427_ml-with-all-items-20210821225639_get-bros_1884175901915086848_bros_univ_dataset'

path = os.path.join(folder, file)
path2 = os.path.join(folder, file2)


# cal_KRX = get_calendar('XKRX')  

# def get_krx_on_dates_n_days_ago(date_ref, n_days=20):
#     return [date.strftime('%Y%m%d')
#             for date in pd.bdate_range(
#         end=date_ref, freq='C', periods=n_days,
#         holidays=cal_KRX.precomputed_holidays) ]

#df_market_info 가져오기
df_market = pd.read_csv(path,
                        index_col=0,
                        dtype={'날짜':str, '등락률': 'double'}
                        ).reset_index(drop=True)

dates_in_set = df_market.날짜.unique().tolist()
dates_on_train = df_market.날짜.unique().tolist()[-20:]

#df_ed 가져오기
df_ed = pd.read_csv(path2, index_col=0).reset_index(drop=True)
df_ed_r = df_ed.copy() 
df_ed_r.rename(columns={'target':'source', 'source':'target'}, inplace=True)
df_ed2 = df_ed.append(df_ed_r, ignore_index=True)
df_ed2['date'] = pd.to_datetime(df_ed2.date).dt.strftime('%Y%m%d')

#%%
df_market.head()
# %%
#functions
@functools.lru_cache()
def get_n_bro_list(code, period, date_ref):
    # 시간이 많이 걸리는 곳
    l_bros = df_ed2[(df_ed2.source == code) & (df_ed2.date == date_ref) & (df_ed2.period == period)].target.to_list()
    # print('l_bros', l_bros)
    return l_bros

def get_df_date_ref(df, date_ref):
    df_ = df[df.날짜 == date_ref]
    return df_

def get_up_bro_ratio(code, period, date_ref): # 친구들 중 오른 친구 비율 /  opts =  60일, 90일, 120일      
    l_bros = get_n_bro_list(code, period, date_ref)
    df_date_ref = get_df_date_ref
    df__ = df_market[df_market.종목코드.isin(l_bros)].등락률 > 0
    # print('shape_of_friends', df__.shape[0])
    ratio_up = df__.sum()  /df__.shape[0]
    if np.isnan(ratio_up):
        ratio_up = 0
    # print('ratio_up', ratio_up)
    return ratio_up

def get_n_bro(code, period, date_ref): # 해당 코드의 친구 수 /  opts =  60일, 90일, 120일 
    l_bros = get_n_bro_list(code, period, date_ref)        
    return len(l_bros)

def get_bro_up_mean(code, period, date_ref): # 오른 친구들만 골라서 평균 얼마나 올랐는지 /  opts =  60일, 90일, 120일
    '''오른 종목의 상승률'''
    l_bros = get_n_bro_list(code, period, date_ref)
    df__= df_market[df_market.종목코드.isin(l_bros)]
    up_mean = df__[df__.등락률 > 0].등락률.mean()
    if np.isnan(up_mean):
        return 0
    return up_mean

def high_close_ratio(df) : # 당일 고가 / 종가의 비
    try :
        h_c_ratio = df.고가 / df.현재가
    except Exception as e:
        h_c_ratio = 0
    return h_c_ratio

def low_close_ratio(df) :
    try :
        l_c_ratio = df.고가 / df.현재가
    except Exception as e:
        l_c_ratio = 0
    return l_c_ratio

def bro_earn_avg(code, period, date_ref): # 모든 친구들의 상승률 평균
    l_bros = get_n_bro_list(code, period, date_ref)
    df_bros = df_market[df_market.종목코드.isin(l_bros)]
    earn_avg = df_bros.등락률.mean()
    if np.isnan(earn_avg):
        return 0
    return earn_avg

def get_volume_change_wrt_10_avg(code, vol, date_ref):
    index0 = dates_in_set.index(date_ref)
    date_from = dates_in_set[index0 - 10]

    df_ = df_market[df_market.종목코드 == code]
    df__ = df_[(df_.날짜 >= date_from) & (df_.날짜 < date_ref)]
    try :
        vol_avg = df__.거래량.mean()
        vol_chg = (vol/vol_avg)
    except :
        vol_chg = 1
    return vol_chg

def get_volume_change_wrt_10_max(code, vol, date_ref):
    index0 = dates_in_set.index(date_ref)
    date_from = dates_in_set[index0 - 10]

    df_ = df_market[df_market.종목코드 == code]
    df__ = df_[(df_.날짜 >= date_from) & (df_.날짜 < date_ref)]
    try :
        vol_max = df__.거래량.max()
        vol_chg = (vol/vol_max)
    except :
        vol_chg = 1
    return vol_chg

def count_top30_n_days(code, date_ref, period):
    
    index0 = dates_in_set.index(date_ref)
    date_from = dates_in_set[index0 - period]    
    df__ = df_market[(df_market.날짜 >= date_from) & (df_market.날짜 <= date_ref)]
    l_top30 = []
    for date, df in df__.groupby('날짜') :
        df_temp = df.sort_values(by='등락률',  ascending=False)
        l_top30.extend(df_temp.head(30).종목코드.to_list())
    c_result = Counter(l_top30)
    try:
        c = c_result[code]
    except :
        c = 0
    return c 
# %%
df_ = df_market[df_market.날짜.isin(dates_on_train)].head(100)
df_ = df_[['순위_상승률', '순위_시가총액', '거래대금', '거래량', '고가', '날짜', '등락률', '외국인비율', '저가', '전일거래량', '종목명', '종목코드', '현재가']]
df_['up_bro_ratio_120'] = df_.apply(lambda row: get_up_bro_ratio(row.종목코드, 120, row.날짜), axis=1)
df_['n_bros_120'] = df_.apply(lambda row: get_n_bro(row.종목코드, 120, row.날짜), axis=1)
df_['up_bros_mean_120'] = df_.apply(lambda row: get_bro_up_mean(row.종목코드, 120, row.날짜), axis=1)

df_['up_bro_ratio_90'] = df_.apply(lambda row: get_up_bro_ratio(row.종목코드, 90, row.날짜), axis=1)
df_['n_bros_90'] = df_.apply(lambda row: get_n_bro(row.종목코드, 90, row.날짜), axis=1)
df_['up_bros_mean_90'] = df_.apply(lambda row: get_bro_up_mean(row.종목코드, 90, row.날짜), axis=1)

df_['up_bro_ratio_60'] = df_.apply(lambda row: get_up_bro_ratio(row.종목코드, 60, row.날짜), axis=1)
df_['n_bros_60'] = df_.apply(lambda row: get_n_bro(row.종목코드, 60, row.날짜), axis=1)
df_['up_bros_mean_60'] = df_.apply(lambda row: get_bro_up_mean(row.종목코드, 60, row.날짜), axis=1)

df_['up_bro_ratio_40'] = df_.apply(lambda row: get_up_bro_ratio(row.종목코드, 40, row.날짜), axis=1)
df_['n_bros_40'] = df_.apply(lambda row: get_n_bro(row.종목코드, 40, row.날짜), axis=1)
df_['up_bros_mean_40'] = df_.apply(lambda row: get_bro_up_mean(row.종목코드, 40, row.날짜), axis=1)

df_['up_bro_ratio_20'] = df_.apply(lambda row: get_up_bro_ratio(row.종목코드, 20, row.날짜), axis=1)
df_['n_bros_20'] = df_.apply(lambda row: get_n_bro(row.종목코드, 20, row.날짜), axis=1)
df_['up_bros_mean_20'] = df_.apply(lambda row: get_bro_up_mean(row.종목코드, 20, row.날짜), axis=1)

df_['h_c_ratio'] = df_.apply(lambda row: high_close_ratio(row), axis=1)
df_['l_c_ratio'] = df_.apply(lambda row: low_close_ratio(row), axis=1)

df_['bro_earn_avg_120'] = df_.apply(lambda row : bro_earn_avg(row.종목코드, 120, row.날짜), axis=1)
df_['bro_earn_avg_90'] = df_.apply(lambda row : bro_earn_avg(row.종목코드, 90, row.날짜), axis=1)
df_['bro_earn_avg_60'] = df_.apply(lambda row : bro_earn_avg(row.종목코드, 60, row.날짜), axis=1)
df_['bro_earn_avg_40'] = df_.apply(lambda row : bro_earn_avg(row.종목코드, 40, row.날짜), axis=1)
df_['bro_earn_avg_20'] = df_.apply(lambda row : bro_earn_avg(row.종목코드, 20, row.날짜), axis=1)

df_['top30_count_10days'] = df_.apply(lambda row : count_top30_n_days(row.종목코드, row.날짜, 10), axis=1)
df_['top30_count_5days'] = df_.apply(lambda row : count_top30_n_days(row.종목코드, row.날짜, 5), axis=1)

df_['volume_change_wrt_10_avg'] = df_.apply(lambda row:get_volume_change_wrt_10_avg(row.종목코드, row.거래량, row.날짜), axis=1)
df_['volume_change_wrt_10_max'] = df_.apply(lambda row:get_volume_change_wrt_10_max(row.종목코드, row.거래량, row.날짜), axis=1)

df_.fillna(0, inplace=True)
# df_.to_csv(features_dataset.path)

# %%
%%timeit -n1
df_short = df_.head(100)

l_up_bro_ratio_120 = []
l_up_bro_ratio_90 = []
l_up_bro_ratio_60 = []
l_up_bro_ratio_40 = []
l_up_bro_ratio_20 = []

n_bros_120 = []
n_bros_90 = []
n_bros_60 = []
n_bros_40 = []
n_bros_20 = []

for idx, row in df_short.iterrows():
    l_up_bro_ratio_120.append(get_up_bro_ratio(row.종목코드, 120, row.날짜))
    l_up_bro_ratio_90.append(get_up_bro_ratio(row.종목코드, 90, row.날짜))
    l_up_bro_ratio_60.append(get_up_bro_ratio(row.종목코드, 60, row.날짜))
    l_up_bro_ratio_40.append(get_up_bro_ratio(row.종목코드, 40, row.날짜))
    l_up_bro_ratio_20.append(get_up_bro_ratio(row.종목코드, 20, row.날짜))

    n_bros_120.append(get_n_bro(row.종목코드, 120, row.날짜))
    n_bros_90.append(get_n_bro(row.종목코드, 90, row.날짜))
    n_bros_60.append(get_n_bro(row.종목코드,60, row.날짜))
    n_bros_40.append(get_n_bro(row.종목코드, 40, row.날짜))
    n_bros_20.append(get_n_bro(row.종목코드,20, row.날짜))

# %%
