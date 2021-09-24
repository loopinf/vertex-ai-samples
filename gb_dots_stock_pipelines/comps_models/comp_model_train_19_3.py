from datetime import date
from kfp.components import InputPath, OutputPath
from typing import NamedTuple
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)

def train_model_19_3(
  ml_dataset : Input[Dataset],
  bros_univ_dataset: Input[Dataset],
  predict_dataset: Output[Dataset],
  model01 : Output[Model],
  model02 : Output[Model],
  model03 : Output[Model],
#   predictions_path : OutputPath('DataFrame')
) -> NamedTuple(
    'Outputs',
    [ ('ver', str)  
]):

    ver = '19_3'

    import collections
    import pandas as pd
    import numpy as np
    import FinanceDataReader as fdr
    import os

    df_dataset = pd.read_pickle(ml_dataset.path)
    df_preP = df_dataset.copy()
    print(f'size01 {df_preP.shape}')

    # bros
    df_bros = pd.read_pickle(bros_univ_dataset.path)
    df_bros = df_bros[df_bros.period.isin(['60', '90', '120'])]

    # drop duplicated column
    cols_ohlcv_x = ['open_x', 'high_x', 'low_x', 'close_x', 'volume_x', 'change_x']
    cols_ohlcv_y = ['open_y', 'high_y', 'low_y', 'close_y', 'volume_y']

    df_preP.rename(columns={"change_y" : "change"}, inplace=True)

    df_preP = df_preP.drop(columns=cols_ohlcv_x+cols_ohlcv_y)

    print(f'size02 {df_preP.shape}')

    # drop SPACs
    stock_names = pd.Series(df_preP.name.unique())
    stock_names_SPAC = stock_names[ stock_names.str.contains('스팩')].tolist()

    df_preP = df_preP.where( 
                lambda df : ~df.name.isin(stock_names_SPAC)
                ).dropna(subset=['name'])

    print(f'size03 {df_preP.shape}')

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
    
    print(f'size04 {df_preP.shape}')

    df_preP['date'] = pd.to_datetime(df_preP.date)
    df_preP['admin_stock'] = df_preP.DesignationDate <= df_preP.date
    df_preP = (
            df_preP.where(
                lambda df: df.admin_stock == 0
            ).dropna(subset=['admin_stock'])
            ) 
    print(f'size05 {df_preP.shape}')
    # Add day of week
    df_preP['dayofweek'] = pd.to_datetime(df_preP.date.astype('str')).dt.dayofweek.astype('category')
    
    df_preP['mkt_cap_cat'] = pd.cut(
                                df_preP['mkt_cap'],
                                bins=[0, 100, 2000, 5000, 10000, np.inf],
                                include_lowest=True,
                                labels=['A', 'B', 'C', 'D', 'E'])

    # Set Target & Features
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
            # 'in_top_30_5',
            # 'in_top_30_10',
            # 'in_top_30_20',
            # 'up_bro_ratio_20',
            # 'up_bro_ratio_40',
            # 'up_bro_ratio_60',
            # 'up_bro_ratio_90',
            # 'up_bro_ratio_120',
            # 'n_bro_20',
            # 'n_bro_40',
            # 'n_bro_60',
            # 'n_bro_90',
            # 'n_bro_120',
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

    # Change datetime format to str
    df_preP['date'] = df_preP.date.dt.strftime('%Y%m%d')

    date_ref = df_preP.date.max()

    # Extract dataframe for train : 
    dates_for_train = sorted(df_preP.date.unique())[-23:-3]
    

    # Get df_univ for training : top30 & friends for everyday, sum all of these
    def get_15pct_univ_in_period(df, l_dates): # input dataframe : top30s in the period

        print(f'length of l_date : {l_dates.__len__()}')
        df_univ = pd.DataFrame()

        for date in l_dates :
            df_of_the_day = df[df.date == date]
            
            df_15pct_of_the_day = df_of_the_day[(df_of_the_day.change >= -0.15) & (df_of_the_day.change <= 0.15)]
            
            df_ = df_15pct_of_the_day

            df_univ = df_univ.append(df_)
            print(f'check_size_{date}_:_{df_univ.shape}')
            print(f'max_change : {df_univ.change.max()} / min_change : {df_univ.change.min()}')
            print(f'{df_univ.date.unique().tolist()}')
        return df_univ
        
    #
    df_preP = df_preP.drop_duplicates(subset=['date', 'code'])

    df_train = get_15pct_univ_in_period(df_preP, dates_for_train)
    df_train = df_train.dropna(axis=0, subset=target_col)   # target 없는 날짜 제외

    df_pred = get_15pct_univ_in_period(df_preP, [date_ref])
    print(f'check_pred_size : {df_pred.shape}')
    df_pred = df_pred[cols_indicator + features]
    print(f'check_pred_size : {df_pred.shape}')
    df_pred['in_top30'] = df_pred.in_top30.astype('int')
    print(f'check_pred_size : {df_pred.shape}')
    df_pred.to_pickle(predict_dataset.path)

    # ML Model
    from catboost import CatBoostRegressor
    from sklearn.model_selection import train_test_split
    from catboost import Pool
    from catboost.utils import get_roc_curve, get_confusion_matrix

    # Set Model
    model = CatBoostRegressor(
            # random_seed = 42,
            # task_type = 'GPU',
            # iterations=3000,
            iterations=3000,
            train_dir = '/tmp',
            # verbose=500,
            silent=True
        )

    X = df_train[features] 
    y = df_train[target_col].astype('float')
    X['in_top30'] = X.in_top30.astype('int')
   

    # Run prediction 3 times
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

        model.fit(X_train, y_train,
                    use_best_model=True,
                    eval_set = eval_dataset,
                    # , verbose=200
                    # , plot=True, 
                    # cat_features=['in_top30','dayofweek', 'mkt_cap_cat']
                    cat_features=['in_top30']
                    )

        print(f'model score : {model.score(X_test, y_test)}')
        model_path = f"/gcs/pipeline-dots-stock/bong_model/bong_model_{ver}_{date_ref}_{iter_n}"

        model.save_model(model_path)
        if iter_n == 0:
            model.save_model(model01.path)
        if iter_n == 1:
            model.save_model(model02.path)
        if iter_n == 2:
            model.save_model(model03.path)
    outputs = collections.namedtuple(
        "Outputs",
        ["ver"]
    )
    return outputs(ver)