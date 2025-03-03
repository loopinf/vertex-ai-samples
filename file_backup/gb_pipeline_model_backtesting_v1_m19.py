'''
Author : SH KIM
Date : 2021-09-17

Train Univ : +-15% / 20days
Pred Univ : +-15% / 1day

Target : 'change_p1'

Model Type : Regressor
'''

job_file_name='gb-model-backtesting-m19-reg-15pct-univ.json'


# -*- coding: utf-8 -*-
import sys
import os

from catboost.core import CatBoostRegressor
# import pandas as pd

PROJECT_ID = "dots-stock"  # @param {type:"string"}
REGION = "us-central1"  # @param {type:"string"}
USER = "shkim01"  # <---CHANGE THIS
BUCKET_NAME = "gs://pipeline-dots-stock"  # @param {type:"string"}
PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root/{USER}"

from typing import NamedTuple

from kfp import dsl
from kfp.v2 import compiler
import kfp.components as comp
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics,
                        component)
from kfp.v2.google.client import AIPlatformClient

@component(
    base_image="gcr.io/dots-stock/python-img-v5.2",
    packages_to_install=['catboost', 'scikit-learn', 'ipywidgets']
)
def model_backtesting(surfix : str) -> NamedTuple(
'Outputs',
[('surfix', str),
('Return_sum_period',float),
('Num_of_Predicted_day', int),
('Max_return', float),
('Min_return', float),
('Period', int)
]):
        

    import pandas as pd
    import numpy as np

    import FinanceDataReader as fdr
    import pickle

    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split
    from catboost import Pool
    from catboost.utils import get_roc_curve, get_confusion_matrix

    # import os
    # import shap

    #%%
    # #2 Loading Files
    ml_dataset = '/gcs/pipeline-dots-stock/ml_dataset/ml_dataset_20210914_260.pkl'
    bros_dataset = '/gcs/pipeline-dots-stock/ml_dataset/bros_dataset_20210914_260'

    df_ml_dataset = pd.read_pickle(ml_dataset)
    df_bros_dataset = pd.read_pickle(bros_dataset)

    #%%
    # #3 Dataframe Copy 
    df_preP = df_ml_dataset.copy()
    df_bros = df_bros_dataset.copy()

    #%%
    # #4 Pre-processing

    cols_ohlcv_x = ['open_x', 'high_x', 'low_x', 'close_x', 'volume_x']
    cols_ohlcv_y = ['open_y', 'high_y', 'low_y', 'close_y', 'volume_y', 'change_y']

    df_preP.rename(columns={"change_x" : "change"}, inplace=True)

    df_preP = df_preP.drop(columns=cols_ohlcv_x+cols_ohlcv_y)

    # drop SPACs
    stock_names = pd.Series(df_preP.name.unique())
    stock_names_SPAC = stock_names[ stock_names.str.contains('스팩')].tolist()

    df_preP = df_preP.where( 
                lambda df : ~df.name.isin(stock_names_SPAC)
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
    # #5 Set preferences

    # Dates things ...
    l_dates = df_preP.date.unique().tolist()
    idx_start = l_dates.index('20210802')

    period = int(l_dates.__len__() - idx_start)

    # Filtering functions
    def get_15pct_univ_in_period(df, l_dates): # input dataframe : top30s in the period

        print(f'length of l_date : {l_dates.__len__()}')
        df_univ = pd.DataFrame()

        for date in l_dates :
            df_of_the_day = df[df.date == date]
            
            df_15pct_of_the_day = df_of_the_day[(df_of_the_day.change >= -15) & (df_of_the_day.change <= 15)]
            
            l_15pcts_in_date = df_15pct_of_the_day.code.to_list()
            
            df_bros_in_date = df_bros[df_bros.date == date]

            l_bros_of_top30s = df_bros_in_date[\
                    df_bros_in_date.source.isin(l_15pcts_in_date)].target.unique().tolist()

            df_bros_of_top30 = df_of_the_day[df_of_the_day.code.isin(l_bros_of_top30s)]

            df_ = df_15pct_of_the_day.append(df_bros_of_top30)
            
            df_.drop_duplicates(subset=['code'], inplace=True)

            df_univ = df_univ.append(df_)
        # print('size of returned :', df_univ.shape)
        return df_univ

    #%%
    # #6 Set Target and Feats

    # target_col = ['target_close_over_10']
    target_col = ['change_p1']
    cols_indicator = [ 'code', 'name', 'date', ]

    features = [
            #  'code',
            #  'name',
            #  'date',
            # 'rank',
            'mkt_cap',
            # 'mkt_cap_cat',
            'in_top30',
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
            'all_bro_rtrn_mean_60',
            'all_bro_rtrn_mean_90',
            'all_bro_rtrn_mean_120',
            # 'up_bro_rtrn_mean_20',
            # 'up_bro_rtrn_mean_40',
            'up_bro_rtrn_mean_60',
            'up_bro_rtrn_mean_90',
            'up_bro_rtrn_mean_120',
            # 'all_bro_rtrn_mean_ystd_20',
            # 'all_bro_rtrn_mean_ystd_40',
            'all_bro_rtrn_mean_ystd_60',
            'all_bro_rtrn_mean_ystd_90',
            'all_bro_rtrn_mean_ystd_120',
            # 'bro_up_ratio_ystd_20',
            # 'bro_up_ratio_ystd_40',
            'bro_up_ratio_ystd_60',
            'bro_up_ratio_ystd_90',
            'bro_up_ratio_ystd_120',
            # 'up_bro_rtrn_mean_ystd_20',
            # 'up_bro_rtrn_mean_ystd_40',
            'up_bro_rtrn_mean_ystd_60',
            'up_bro_rtrn_mean_ystd_90',
            'up_bro_rtrn_mean_ystd_120',
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
            'return_lag_1',
            'return_lag_2',
            'return_lag_3',
            'bb_u_ratio',
            'bb_l_ratio',
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



    #%%
    # #7 Run Backtesting
    df_pred_all = pd.DataFrame()
    dic_model = {}
    dic_pred = {}
    for i in range(idx_start, l_dates.__len__()):

        dates_for_train = l_dates[i-23: i-3] # 며칠전까지 볼것인가!! 20일만! 일단은
        date_ref = l_dates[i]

        print(f'train date :  from {dates_for_train[0]} to {dates_for_train[-1]}')
        print(f'prediction date : {date_ref}')

        df_train = get_15pct_univ_in_period(df_preP, dates_for_train)
        df_train = df_train.dropna(axis=0, subset=target_col) 

        # Prediction Dataset Concept used by mistake
        df_pred = get_15pct_univ_in_period(df_preP, [date_ref])
        df_pred['date'] = date_ref

        dic_pred[f'{date_ref}'] = df_pred[features] # df_pred 모아두기

        # ML Model
        from catboost import CatBoostClassifier, CatBoostRegressor
        from sklearn.model_selection import train_test_split
        from catboost import Pool
        from catboost.utils import get_roc_curve, get_confusion_matrix

        model_01 = CatBoostRegressor(
                iterations=1500,
                train_dir = '/tmp',
                # verbose=500,
                silent=True
            )

        X = df_train[features + cols_indicator]
        y = df_train[target_col].astype('float')
        

        # Run prediction 3 times
        df_pred_final_01 = pd.DataFrame()
        for iter_n in range(3):

            X_train, X_test, y_train, y_test = train_test_split(X, y)

            X_train = X_train[features]
            X_test = X_test[features]

            eval_dataset = Pool(
                    X_test, y_test,
                    # cat_features=['mkt_cap_cat']
                    cat_features=['in_top30']
                    )

            print('X Train Size : ', X_train.shape, 'Y Train Size : ', y_train.shape)
            print('No. of true : ', y_train.sum() )

            model_01.fit(X_train, y_train,
                        use_best_model=True,
                        eval_set = eval_dataset,
                        # , verbose=200
                        # , plot=True, 
                        # cat_features=['in_top30','dayofweek', 'mkt_cap_cat']
                        cat_features=['in_top30']
                        )

            dic_model[f'{date_ref}_{iter_n}'] = model_01

            print(f'model score : {model_01.score(X_test, y_test)}')

            # Prediction
            pred_stocks_01 = model_01.predict(df_pred[features])  
            
            df_pred_stocks_01 = pd.DataFrame(pred_stocks_01, columns=['Prediction']).reset_index(drop=True)

            df_pred_name_code = df_pred[cols_indicator].reset_index(drop=True)

            df_pred_r_01 = pd.concat(
                            [
                            df_pred_name_code,
                            df_pred_stocks_01,
                            ],
                            axis=1)

            df_pred_r_01 = df_pred_r_01[df_pred_r_01.Prediction > 0]
            print(f'iter_number_{iter_n}')
            df_pred_final_01 = df_pred_final_01.append(df_pred_r_01)

        df_pred_final_01 = df_pred_final_01.groupby(['name', 'code', 'date']).mean() # apply mean to duplicated recommends
        df_pred_final_01 = df_pred_final_01.reset_index()
        df_pred_final_01 = df_pred_final_01.sort_values(by='Prediction', ascending=False) # high probability first
        
        df_pred_final_01.drop_duplicates(subset=['code', 'date'], inplace=True) 

        df_pred_all = df_pred_all.append(df_pred_final_01)
        print(f'size of df_pred_all : {df_pred_all.shape}' )


    # Save dic_model / dic_df_pred
    path_dic_model = f'/gcs/pipeline-dots-stock/gb_backtesting_results/dic_model_{surfix}'
    path_dic_df_pred = f'/gcs/pipeline-dots-stock/gb_backtesting_results/dic_df_pred_{surfix}'

    with open(path_dic_model, 'wb') as f:
        pickle.dump(dic_model, f)

    with open(path_dic_df_pred, 'wb') as f:
        pickle.dump(dic_pred, f)

    #%%
    # #8 Update Price for prediction result
    
    date_start = sorted(df_pred_all.date.unique())[0]
    codes_to_update = df_pred_all.code.unique().tolist()

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
        df_['c_1'] = df_.close.shift(-1)
        df_['c_2'] = df_.close.shift(-2)
        df_['c_3'] = df_.close.shift(-3)

        return df_

    df_price_updated  = df_price.groupby('code').apply(lambda df: get_price_tracked(df))
    df_price_updated = df_price_updated[['date', 'code', 'c_1', 'c_2', 'c_3', 'close']]
    df_price_updated = df_price_updated.reset_index(drop=True)

    df_price_updated = df_pred_all.merge(
                            df_price_updated,
                            left_on=['date', 'code'],
                            right_on=['date', 'code'] )

    df_price_updated.dropna(inplace=True)

    # Calc daily return in %

    def calc_return(df):
        r1 = (df.c_1 / df.close - 1) * 100
        r1 = format(r1, '.1f')

        r2 = (df.c_2 / df.close - 1) * 100
        r2 = format(r2, '.1f')

        r3 = (df.c_3 / df.close - 1) * 100
        r3 = format(r3, '.1f')

        df['r1'] = float(r1)
        df['r2'] = float(r2)
        df['r3'] = float(r3)

        return df

    df_return_updated = df_price_updated.apply(lambda row: calc_return(row), axis=1)

    # Save return calculated df
    df_return_updated.to_pickle(f'/gcs/pipeline-dots-stock/gb_backtesting_results/df_return_updated_{surfix}')

    # %%
    # #9 Apply sell condition and calc final return

    def return_final(df):

        df['f_r'] = df.r1

        return df


    df_sell_condi = df_return_updated.apply(lambda row: return_final(row), axis=1)
    # %%
    # #10 Calc daily return 

    daily_return = []

    def calc_daily_return(df):
        df_ = df.sort_values(by='Prediction', ascending=False)
        df_ = df.head(10)
        
        rr = df_.f_r.mean()
        daily_return.append(rr)
        print(rr)
        
    df_sell_condi.groupby('date').apply(lambda df : calc_daily_return(df))

    #%%
    # 11 Calc sum
    sum_of_period = float(sum(daily_return))
    num_of_p_day = int(daily_return.__len__())
    min_r = float(min(daily_return))
    max_r = float(max(daily_return))

    return (surfix, sum_of_period, num_of_p_day, max_r, min_r, period)

# create pipeline 
#########################################

@dsl.pipeline(
  name=job_file_name.split('.json')[0],
  pipeline_root=PIPELINE_ROOT
)    
def we_would_be_gb_in_this_year():

    op_model_backtesting = model_backtesting('m19_regressor_15pct_univ_06')

compiler.Compiler().compile(
  pipeline_func=we_would_be_gb_in_this_year,
  package_path=job_file_name
)

api_client = AIPlatformClient(
    project_id=PROJECT_ID,
    region=REGION,
)

response = api_client.create_run_from_job_spec(
  job_spec_path=job_file_name,
  enable_caching= False,
  pipeline_root=PIPELINE_ROOT
)