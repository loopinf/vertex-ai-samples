from kfp.components import InputPath, OutputPath
from typing import NamedTuple
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)

def get_model_backtesting(
  ml_dataset : Input[Dataset],
  bros_univ_dataset: Input[Dataset],
  predictions: Output[Dataset]
):
    ver = '10'

    import pandas as pd
    import numpy as np
    from pykrx import stock
    import FinanceDataReader as fdr
    import os
    import pickle

    df_dataset = pd.read_pickle(ml_dataset.path)
    df_preP = df_dataset.copy()
    print(f'size01 {df_preP.shape}')

    df_bros = pd.read_pickle(bros_univ_dataset.path)
    df_bros = df_bros[df_bros.period.isin(['60', '90', '120'])]

    # drop duplicated column
    cols_ohlcv_x = ['open_x', 'high_x', 'low_x', 'close_x', 'volume_x', 'change_x']
    cols_ohlcv_y = ['open_y', 'high_y', 'low_y', 'close_y', 'volume_y', 'change_y']
    df_preP = df_preP.drop(columns=cols_ohlcv_x+cols_ohlcv_y)

    print(f'size02 {df_preP.shape}')

    # drop SPACs
    stock_names = pd.Series(df_preP.name.unique())
    stock_names_SPAC = stock_names[ stock_names.str.contains('스팩')].tolist()

    df_preP = df_preP.where( 
                lambda df : ~df.name.isin(stock_names_SPAC)
                ).dropna(subset=['name'])

    print(f'size03 {df_preP.shape}')

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

    # Add market_cap categotu
    df_preP['mkt_cap_cat'] = pd.cut(
                                df_preP['mkt_cap'],
                                bins=[0, 1000, 5000, 10000, 50000, np.inf],
                                include_lowest=True,
                                labels=['A', 'B', 'C', 'D', 'E'])

    # Set Target & Features
    target_col = ['target_close_over_10']
    cols_indicator = [
                        'code',
                        'name',
                        'date',
                        ]

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
             'close_30_sma',
             'close_60_sma',
            #  'daily_return',
            'return_lag_1',
            'return_lag_2',
            'return_lag_3',
            'bb_u_ratio',
            'bb_l_ratio',
            # 'max_scale_MACD',
            'volume_change_wrt_10max',
            'volume_change_wrt_5max',
            # 'volume_change_wrt_20max',
            'volume_change_wrt_10mean',
            'volume_change_wrt_5mean',
            # 'volume_change_wrt_20mean',
            # 'close_ratio_wrt_10max',
            # 'close_ratio_wrt_10min',
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

    # Split Dataset into For Training & For Prediction
    l_dates = df_preP.date.unique().tolist()
    idx_start = l_dates.index('20210802')

    # Filtering function
    def get_univ_bh01(df, l_dates): # input dataframe : top30s in the period

        # l_dates = df.date.unique().tolist()
        print(f'length of l_date : {l_dates.__len__()}')
        df_univ = pd.DataFrame()
        for date in l_dates :
            df_of_the_day = df[df.date == date]
            df_of_the_day = df_of_the_day.sort_values(by='rank', ascending=True)
            # print('df_of_the_date', df_of_the_day)
            df_top30_in_date = df_of_the_day.head(30)
            l_top30s_in_date = df_top30_in_date.code.to_list()
            # print(f'size of top30 in the date : {df_top30_in_date.shape}')
            
            df_bros_in_date = df_bros[df_bros.date == date]
            l_bros_of_top30s = df_bros_in_date[\
                    df_bros_in_date.source.isin(l_top30s_in_date)].target.unique().tolist()
            df_bros_of_top30 = df_of_the_day[df_of_the_day.code.isin(l_bros_of_top30s)]
            # print(f'size of top30 + friends in the date : {df_bros_of_top30.shape}')

            df_ = df_top30_in_date.append(df_bros_of_top30)
            # print('size of the day : ', df_.shape)
            df_univ = df_univ.append(df_)
        # print('size of returned :', df_univ.shape)
        return df_univ

    df_pred_all = pd.DataFrame()
    for i in range(idx_start, l_dates.__len__()):

        dates_for_train = l_dates[i-23: i-3] # 며칠전까지 볼것인가!! 20일만! 일단은
        dates_for_pred = l_dates[i-9:i+1]  # prediction date
        date_ref = dates_for_pred[-1]

        print(f'train date :  from {dates_for_train[0]} to {dates_for_train[-1]}')
        print(f'prediction date : from {dates_for_pred[0]} to {dates_for_pred[-1]}')

        df_train = get_univ_bh01(df_preP, dates_for_train)
        df_pred = get_univ_bh01(df_preP, dates_for_pred)
        df_pred['date'] = date_ref

        # df_train = df_preP[df_preP.date.isin(dates_for_train)]
        df_train = df_train.dropna(axis=0, subset=target_col)   # target 없는 날짜 제외

        # df_pred = df_preP[df_preP.date == date_for_pred] 

        # print(f'check01 : train size {df_train.shape} / pred size {df_pred.shape}')
        # ML Model
        from catboost import CatBoostClassifier
        from sklearn.model_selection import train_test_split
        from catboost import Pool
        from catboost.utils import get_roc_curve, get_confusion_matrix

        # Set Model
        model_01 = CatBoostClassifier(
                # random_seed = 42,
                # task_type = 'GPU',
                # iterations=3000,
                iterations=3000,
                train_dir = '/tmp',
                # verbose=500,
                silent=True
            )

        X = df_train[features + cols_indicator]
        y = df_train[target_col].astype('float')
        # X['in_top30'] = X.in_top30.astype('int')
        # df_pred['in_top30'] = df_pred.in_top30.astype('int')

        # Run prediction 3 times
        df_pred_final_01 = pd.DataFrame()
        for iter_n in range(3):

            X_train, X_test, y_train, y_test = train_test_split(X, y)

            X_train_indictor = X_train[cols_indicator]
            X_test_indictor = X_test[cols_indicator]

            X_train = X_train[features]
            X_test = X_test[features]

            eval_dataset = Pool(
                    X_test, y_test,
                    # cat_features=['mkt_cap_cat']
                    # cat_features=['in_top30']
                    )

            print('X Train Size : ', X_train.shape, 'Y Train Size : ', y_train.shape)
            print('No. of true : ', y.sum() )

            model_01.fit(X_train, y_train,
                        # use_best_model=True,
                        # eval_set = eval_dataset,
                        # , verbose=200
                        # , plot=True, 
                        # cat_features=['in_top30','dayofweek', 'mkt_cap_cat']
                        # cat_features=['in_top30']
                        )

            print(f'model score : {model_01.score(X_test, y_test)}')
            model_folder = "/gcs/pipeline-dots-stock/bong_model"
            model_name = f'bong_model_{ver}_{date_ref}_{iter_n}'
            path = os.path.join(model_folder, model_name)

            model_01.save_model(path)

            # Prediction
            pred_stocks_01 = model_01.predict(df_pred[features])    
            pred_proba_01 = model_01.predict_proba(df_pred[features])
            
            df_pred_stocks_01 = pd.DataFrame(pred_stocks_01, columns=['Prediction']).reset_index(drop=True)
            df_pred_proba_01 = pd.DataFrame(pred_proba_01, columns=['Proba01', 'Proba02']).reset_index(drop=True)

            df_pred_name_code = df_pred[cols_indicator].reset_index(drop=True)

            df_pred_r_01 = pd.concat(
                            [
                            df_pred_name_code,
                            df_pred_stocks_01,
                            df_pred_proba_01
                            ],
                            axis=1)

            df_pred_r_01 = df_pred_r_01[df_pred_r_01.Prediction > 0]
            # print('prediction results', df_pred_r_01)
            df_pred_final_01 = df_pred_final_01.append(df_pred_r_01)

        # print(f'size of df_pred_final_01_1 : {df_pred_final_01.shape}' )
        df_pred_final_01 = df_pred_final_01.groupby(['name', 'code', 'date']).mean() # apply mean to duplicated recommends
        df_pred_final_01 = df_pred_final_01.reset_index()
        df_pred_final_01 = df_pred_final_01.sort_values(by='Proba02', ascending=False) # high probability first
        # print(f'size of df_pred_final_01_2 : {df_pred_final_01.shape}' )

        df_pred_final_01.drop_duplicates(subset=['code', 'date'], inplace=True) # remove duplicates
        # print(f'size of df_pred_final_01_3 : {df_pred_final_01.shape}' )

        df_pred_all = df_pred_all.append(df_pred_final_01)
        print(f'size of df_pred_all : {df_pred_all.shape}' )

    prediction_folder = '/gcs/pipeline-dots-stock/bong_predictions'
    prediction_name = f'bong_{ver}.pkl'
    path_pred = os.path.join(prediction_folder, prediction_name)

    with open(path_pred, 'wb') as f:
        pickle.dump(df_pred_all, f)

    df_pred_all.to_pickle(predictions.path) # save
