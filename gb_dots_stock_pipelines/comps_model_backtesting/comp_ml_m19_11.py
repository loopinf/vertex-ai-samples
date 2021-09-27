from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)

def get_ml_op(
    start_date : str,
    pre_processed_dataset : Input[Dataset],
    bros_dataset : Input[Dataset],
    dic_model_dataset : Output[Dataset],
    dic_df_pred_dataset : Output[Dataset],
    prediction_result_dataset : Output[Dataset]
) -> str :
    
    DESC = "model m19-11 is Classifier over1 no bros / include top30 / include KODEX ETN / All items for Prediction / 15% for Training"

    import pandas as pd
    import pickle

    from sklearn.model_selection import train_test_split
    from catboost import Pool
    from catboost import CatBoostClassifier, CatBoostRegressor  
    
    # Load Dataset
    df_preP = pd.read_pickle(pre_processed_dataset.path)
    df_bros = pd.read_pickle(bros_dataset.path)

    # Dates things ...
    l_dates = df_preP.date.unique().tolist()
    print(f'df_preP start from {l_dates[0]} end at {l_dates[-1]} shape : {df_preP.shape}')
    idx_start = l_dates.index(start_date)
    print(f'index of start date : {idx_start}')

    period = int(l_dates.__len__() - idx_start)

    # get Univ df
    def get_15pct_univ_in_period(df, l_dates): # input dataframe : top30s in the period

        print(f'length of l_date : {l_dates.__len__()}')
        df_univ = pd.DataFrame()

        for date in l_dates :
            df_of_the_day = df[df.date == date]            
            df_15pct_of_the_day = df_of_the_day[(df_of_the_day.change >= -0.15) & (df_of_the_day.change <= 0.15)]
            
            # l_codes = df_15pct_of_the_day.code.unique().tolist()

            # df_bros_in_date = df_bros[df_bros.date == date]
            # l_bros_of_top30s = df_bros_in_date[\
            #         df_bros_in_date.source.isin(l_codes)].target.unique().tolist()
            # df_bros_of_top30 = df_of_the_day[df_of_the_day.code.isin(l_bros_of_top30s)]

            df_ = df_15pct_of_the_day #.append(df_bros_of_top30)

            df_.drop_duplicates(subset=['code', 'date'], inplace=True)

            df_univ = df_univ.append(df_)

        return df_univ

   
    # Set Target and Feats

    # target_col = ['target_close_over_10']
    target_col = ['change_p1_over1']
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

    # Model training and prediction
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
        df_pred = df_preP[df_preP.date == date_ref]
        df_pred = df_pred[(df_pred.change >= -0.15) & (df_pred.change <= 0.15)] #get_15pct_univ_in_period(df_preP, [date_ref])
        # df_pred['date'] = date_ref
        print(f'shape of df_pred : {df_pred.shape}')

        dic_pred[f'{date_ref}'] = df_pred[features] # df_pred 모아두기

        # ML Model        
        model = CatBoostClassifier(
                iterations=1000,
                train_dir = '/tmp',
                # verbose=500,
                silent=True
            )

        X = df_train[features + cols_indicator ]
        y = df_train[target_col].astype('float')        

        # Run prediction 3 times
        df_pred_the_day = pd.DataFrame()

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

            model.fit(X_train, y_train,
                        use_best_model=True,
                        eval_set = eval_dataset,
                        # cat_features=['in_top30','dayofweek', 'mkt_cap_cat']
                        cat_features=['in_top30']
                        )

            dic_model[f'{date_ref}_{iter_n}'] = model
            print(model.get_best_iteration())

            # Prediction
            pred_result = model.predict(df_pred[features])
            pred_proba = model.predict_proba(df_pred[features])
            
            df_pred_result = pd.DataFrame(pred_result, columns=['Prediction']).reset_index(drop=True)
            df_pred_proba = pd.DataFrame(pred_proba, columns=['Proba01', 'Proba02']).reset_index(drop=True)
            df_pred_name_code = df_pred[cols_indicator].reset_index(drop=True)

            df_pred_ = pd.concat(
                            [
                            df_pred_name_code,
                            df_pred_result,
                            df_pred_proba,
                            ],
                            axis=1)

            df_pred_ = df_pred_[df_pred_.Prediction > 0]
            df_pred_the_day = df_pred_the_day.append(df_pred_)

            print(f'iter_number_{iter_n}_size_of_df_pred_the_day_{df_pred_the_day.shape}')

        df_pred_the_day = df_pred_the_day.groupby(['name', 'code', 'date']).mean() # apply mean to duplicated recommends
        df_pred_the_day = df_pred_the_day.reset_index()
        # df_pred_the_day = df_pred_the_day.sort_values(by='Prediction', ascending=False) # high probability first
        df_pred_the_day = df_pred_the_day.sort_values(by='Proba02', ascending=False)

        df_pred_the_day.drop_duplicates(subset=['code', 'date'], inplace=True) 

        df_pred_all = df_pred_all.append(df_pred_the_day)
        print(f'size of df_pred_all : {df_pred_all.shape}' )

    with open(dic_model_dataset.path, 'wb') as f:
        pickle.dump(dic_model, f)

    with open(dic_df_pred_dataset.path, 'wb') as f:
        pickle.dump(dic_pred, f)

    df_pred_all.to_pickle(prediction_result_dataset.path)

    return DESC