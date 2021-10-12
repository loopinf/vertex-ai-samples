#%%
import pandas as pd
import numpy as np

import FinanceDataReader as fdr
import pickle

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from catboost import Pool
from catboost.utils import get_roc_curve, get_confusion_matrix


#%%
#2 Loading Files
ml_dataset = 'gs://pipeline-dots-stock/pipeline_root/shkim01/516181956427/gb-create-dataset-for-backtesting-20210925081851/get-ml-dataset_1066060086012542976/ml_dataset'

df_ml_dataset = pd.read_pickle(ml_dataset)

#%%
#3 Dataframe Copy 
df_preP = df_ml_dataset.copy()

#%%
#4 Pre-processing

cols_ohlcv_x = ['open_x', 'high_x', 'low_x', 'close_x', 'volume_x', 'change_x']
cols_ohlcv_y = ['open_y', 'high_y', 'low_y', 'close_y', 'volume_y']

df_preP.rename(columns={"change_y" : "change", 'rank':'Rank'}, inplace=True)

df_preP = df_preP.drop(columns=cols_ohlcv_x+cols_ohlcv_y)

# drop SPACs
stock_names = pd.Series(df_preP.name.unique())
stock_names_SPAC = stock_names[ stock_names.str.contains('스팩')].tolist()

df_preP = df_preP.where( 
            lambda df : ~df.name.isin(stock_names_SPAC)
            ).dropna(subset=['name'])

# drop KODEX
stock_names = pd.Series(df_preP.name.unique())
stock_names_KODEX = stock_names[ stock_names.str.contains('KODEX')].tolist()

df_preP = df_preP.where( 
            lambda df : ~df.name.isin(stock_names_KODEX)
            ).dropna(subset=['name'])

# drop ETN
stock_names = pd.Series(df_preP.name.unique())
stock_names_ETN = stock_names[ stock_names.str.contains('ETN')].tolist()

df_preP = df_preP.where( 
            lambda df : ~df.name.isin(stock_names_ETN)
            ).dropna(subset=['name'])

# Remove administrative items
krx_adm = fdr.StockListing('KRX-ADMINISTRATIVE') # 관리종목
df_preP = df_preP.merge(krx_adm[['Symbol','DesignationDate']], 
    left_on='code', right_on='Symbol', how='left')

df_preP['date'] = pd.to_datetime(df_preP.date)
df_preP['admin_stock'] = df_preP.DesignationDate <= df_preP.date
df_preP = (
            df_preP.where(
                lambda df: df.admin_stock == 0
            ).dropna(subset=['admin_stock'])
            ) 

# Add day of week
df_preP['dayofweek'] = pd.to_datetime(df_preP.date.astype('str')).dt.dayofweek.astype('category')

# Add market_cap categotu
df_preP['mkt_cap_cat'] = pd.cut(
                            df_preP['mkt_cap'],
                            bins=[0, 1000, 5000, 10000, 50000, np.inf],
                            include_lowest=True,
                            labels=['A', 'B', 'C', 'D', 'E'])

# Change datetime format to str
df_preP['date'] = df_preP.date.dt.strftime('%Y%m%d')
df_preP['in_top30'] = df_preP.in_top30.astype('int')

#%%
#5 Set preferences

# Dates things ...
l_dates = df_preP.date.unique().tolist()
idx_start = l_dates.index('20201102')
idx_last = l_dates.__len__() - 1

period = int(l_dates.__len__() - idx_start)

#%%

#6 Filtering functions
def get_top30_univ_for_training(df, l_dates): # input dataframe : top30s in the period

    # l_dates = df.date.unique().tolist()
    print(f'length of l_date : {l_dates.__len__()}')
    df_univ = pd.DataFrame()
    for date in l_dates :
        df_of_the_day = df[df.date == date]
        df_of_the_day = df_of_the_day.sort_values(by='Rank', ascending=True)
        # print('df_of_the_date', df_of_the_day)
        df_top30_in_date = df_of_the_day.head(30)

        df_ = df_top30_in_date
        df_univ = df_univ.append(df_)

    return df_univ

def get_top30_univ_for_pred(df, date):
    print('in')
    df_univ = pd.DataFrame()
    # for date in dates :
    print(f"pred date : {date}")
    df_ = df[df.date == date]
    df_ = df_.sort_values(by='Rank', ascending=True)
    df_ = df_.head(40)
    # df_ = df_[df_.change < 0.29]
    df_univ = df_
    # df_univ = df_univ.append(df_)
    print(f'size of df univ : {df_univ.shape}')
    return df_univ
#%%

#7  
# target_col = ['target_close_over_10']
target_col = ['change_p3_over10'] 
cols_indicator = [ 'code', 'name', 'date', ]

features = [
        #  'code',
        #  'name',
        #  'date',
        # 'rank',
        'mkt_cap',
        # 'mkt_cap_cat',
        # 'in_top30',
        # 'rank_mean_10',
        # 'rank_mean_5',
        'in_top_30_5',
        'in_top_30_10',
        'in_top_30_20',
        # 'up_bro_ratio_20',
        # 'up_bro_ratio_40',
        'up_bro_ratio_60',
        'up_bro_ratio_90',
        'up_bro_ratio_120',
        # 'n_bro_20',
        # 'n_bro_40',
        'n_bro_60',
        'n_bro_90',
        'n_bro_120',
        # 'all_bro_rtrn_mean_20',
        # 'all_bro_rtrn_mean_40',
        # 'all_bro_rtrn_mean_60',
        # 'all_bro_rtrn_mean_90',
        # 'all_bro_rtrn_mean_120',
        # 'up_bro_rtrn_mean_20',
        # 'up_bro_rtrn_mean_40',
        # 'up_bro_rtrn_mean_60',
        # 'up_bro_rtrn_mean_90',
        # 'up_bro_rtrn_mean_120',
        # 'all_bro_rtrn_mean_ystd_20',
        # 'all_bro_rtrn_mean_ystd_40',
        # 'all_bro_rtrn_mean_ystd_60',
        # 'all_bro_rtrn_mean_ystd_90',
        # 'all_bro_rtrn_mean_ystd_120',
        # 'bro_up_ratio_ystd_20',
        # 'bro_up_ratio_ystd_40',
        # 'bro_up_ratio_ystd_60',
        # 'bro_up_ratio_ystd_90',
        # 'bro_up_ratio_ystd_120',
        # 'up_bro_rtrn_mean_ystd_20',
        # 'up_bro_rtrn_mean_ystd_40',
        # 'up_bro_rtrn_mean_ystd_60',
        # 'up_bro_rtrn_mean_ystd_90',
        # 'up_bro_rtrn_mean_ystd_120',
        #  'index',
        #  'open_x',
        #  'high_x',
        #  'low_x',
        #  'close_x',
        #  'volume_x',
        #  'change_x',
        #  'high_p1',
        #  'high_p2',
        #  'high_p3',
        #  'close_p1',
        #  'close_p2',
        #  'close_p3',
        #  'change_p1',
        #  'change_p2',
        #  'change_p3',
        #  'change_p1_over5',
        #  'change_p2_over5',
        #  'change_p3_over5',
        #  'change_p1_over10',
        #  'change_p2_over10',
        #  'change_p3_over10',
        #  'close_high_1',
        #  'close_high_2',
        #  'close_high_3',
        #  'close_high_1_over10',
        #  'close_high_2_over10',
        #  'close_high_3_over10',
        #  'close_high_1_over5',
        #  'close_high_2_over5',
        #  'close_high_3_over5',
        #  'open_y',
        #  'high_y',
        #  'low_y',
        #  'close_y',
        #  'volume_y',
        #  'change_y',
        #  'macd',
        #  'boll_ub',
        #  'boll_lb',
        # 'rsi_30',
        # 'dx_30',
        #  'close_30_sma',
        #  'close_60_sma',
         'daily_return',
        # 'return_lag_1',
        # 'return_lag_2',
        # 'return_lag_3',
        # 'bb_u_ratio',
        # 'bb_l_ratio',
        # 'max_scale_MACD',
        'volume_change_wrt_10max',
        'volume_change_wrt_5max',
        'volume_change_wrt_20max',
        'volume_change_wrt_10mean',
        'volume_change_wrt_5mean',
        'volume_change_wrt_20mean',
        'close_ratio_wrt_10max',
        'close_ratio_wrt_10min',
        'oh_ratio',
        'oc_ratio',
        'ol_ratio',
        'ch_ratio',
        #  'Symbol',
        #  'DesignationDate',
        #  'admin_stock',
        # 'dayofweek'
        ]    

# df_train
df_train = get_top30_univ_for_training(df_preP, l_dates)
df_train = df_train.dropna(axis=0, subset=target_col)

X = df_train[features + cols_indicator]
y = df_train[target_col].astype('float')
print(f'num of true : {y.sum()}')
print(f'num of total size : {X.shape[0]}')

dic_model = {}

# for i_ in range(2):
    
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train = X_train[features]
X_test = X_test[features]


eval_dataset = Pool(
        X_test, y_test,
        # cat_features=['mkt_cap_cat']
        # cat_features=['in_top30']
        )

model = CatBoostClassifier(
        # random_seed = 42,
        # task_type = 'GPU',
        iterations=3000,
        train_dir = '/tmp',
        # verbose=500,
        silent=True
    )

model.fit(X_train, y_train,
                use_best_model=True,
                eval_set = eval_dataset,
                # , verbose=200
                # , plot=True, 
                # cat_features=['in_top30','dayofweek', 'mkt_cap_cat']
                # cat_features=['in_top30']
                )

# dic_model[f'{20211002}_{i_}'] = model

print(f'model score : {model.score(X_test, y_test)}')

#%%
#8


dic_pred = {}
last_idx = l_dates.__len__()

df_pred_all = pd.DataFrame()

for date in l_dates[-100 : -1] :
    print(date)
    idx_date = l_dates.index(date)
    dates = l_dates[idx_date - 5: idx_date]

    df_pred = get_top30_univ_for_pred(df_preP, date)
    # df_pred['date'] = date
    # dic_pred[f'df_pred_{date}'] = df_pred

    df_pred_name_code = df_pred[cols_indicator]
    df_pred_name_code = df_pred_name_code.reset_index(drop=True)

    # df_pred_final = pd.DataFrame()
 
    # for name, model in dic_model.items():
   

    pred_pred = model.predict(df_pred[features])    
    pred_proba = model.predict_proba(df_pred[features])
    
    # print(pred_pred)
    df_pred_r = pd.DataFrame(pred_pred, columns=['Prediction']).reset_index(drop=True)
    df_pred_proba = pd.DataFrame(pred_proba, columns=['Proba01', 'Proba02']).reset_index(drop=True)

    df_pred_r_ = pd.concat(
                    [
                    df_pred_name_code,
                    df_pred_r,
                    df_pred_proba,
                    ],
                    axis=1)

    df_pred_r_ = df_pred_r_[df_pred_r_.Proba02 > 0.4]
        
    df_pred_all = df_pred_all.append(df_pred_r_)
    print(df_pred_all)
    print(df_pred_all.shape)
    # print(f'inter_n : {name}, {df_pred_final.shape}')
    
    # df_pred_final = df_pred_final.groupby(['name', 'code', 'date']).mean()
    # df_pred_final.drop_duplicates(subset=['code', 'date'], inplace=True)

    # df_pred_final = df_pred_final.reset_index()
    # df_pred_final = df_pred_final.sort_values(by='Proba02', ascending=False)    
    
    # df_pred_all = df_pred_all.append(df_pred_final)
    
# path_dic_df_pred = f'/gcs/pipeline-dots-stock/gb_backtesting_results/dic_df_pred_{surfix}'
# with open(path_dic_df_pred, 'wb') as f:
#     pickle.dump(dic_pred, f)

#%%
#9
date_start = sorted(df_pred_all.date.unique())[0]
codes_to_update = df_pred_all.code.unique().tolist()

#%%
def get_price_adj(code, start):
    return fdr.DataReader(code, start=start)    

def get_price(codes, date_start):

    df_price = pd.DataFrame()
    for code in codes :      
        df_ = get_price_adj(code, date_start)
        df_['code'] = code
        df_price = df_price.append(df_)

        # print(df_price.shape, code)      
    return df_price

df_price = get_price(codes_to_update, date_start)
df_price.reset_index(inplace=True)
df_price.columns = df_price.columns.str.lower()
df_price['date'] = df_price.date.dt.strftime('%Y%m%d')

def get_price_tracked(df):

    df_ = df.copy()
    df_.sort_values(by='date', inplace=True)
    df_['c1'] = df_.close.shift(-1)
    df_['c2'] = df_.close.shift(-2)
    df_['c3'] = df_.close.shift(-3)
    df_['c4'] = df_.close.shift(-4)
    df_['c5'] = df_.close.shift(-5)
    df_['c6'] = df_.close.shift(-6)
    df_['c7'] = df_.close.shift(-7)
    df_['c8'] = df_.close.shift(-8)
    df_['c9'] = df_.close.shift(-9)
    df_['c10'] = df_.close.shift(-10)

    return df_

df_price_updated  = df_price.groupby('code').apply(lambda df: get_price_tracked(df))
df_price_updated = df_price_updated[['date', 'code', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'close']]
df_price_updated = df_price_updated.reset_index(drop=True)

df_price_updated = df_pred_all_p.merge(
                        df_price_updated,
                        left_on=['date', 'code'],
                        right_on=['date', 'code'] )

df_price_updated.dropna(inplace=True)

# Calc daily return in %

df_price_updated['r1'] = (df_price_updated['c1'] / df_price_updated['close'] - 1 ) *100
df_price_updated['r2'] = (df_price_updated['c2'] / df_price_updated['close'] - 1 ) *100
df_price_updated['r3'] = (df_price_updated['c3'] / df_price_updated['close'] - 1 ) *100
df_price_updated['r4'] = (df_price_updated['c4'] / df_price_updated['close'] - 1 ) *100
df_price_updated['r5'] = (df_price_updated['c5'] / df_price_updated['close'] - 1 ) *100
df_price_updated['r6'] = (df_price_updated['c6'] / df_price_updated['close'] - 1 ) *100
df_price_updated['r7'] = (df_price_updated['c7'] / df_price_updated['close'] - 1 ) *100
df_price_updated['r8'] = (df_price_updated['c8'] / df_price_updated['close'] - 1 ) *100
df_price_updated['r9'] = (df_price_updated['c9'] / df_price_updated['close'] - 1 ) *100
df_price_updated['r10'] = (df_price_updated['c10'] / df_price_updated['close'] - 1 ) *100

# df_price_updated.to_pickle(f'/gcs/pipeline-dots-stock/gb_backtesting_results/df_price_updated_{surfix}')

daily_return = []
def calc_daily_return(df):
    df_ = df.sort_values(by='Prediction', ascending=False)
    df_ = df.head(10)
    rr = df_.r10.mean()
    daily_return.append(rr)
    print(rr)
    
df_price_updated.groupby('date').apply(lambda df : calc_daily_return(df))

#%%
#7 Calc sum
sum_of_period = float(sum(daily_return))
num_of_p_day = int(daily_return.__len__())
min_r = float(min(daily_return))
max_r = float(max(daily_return))

