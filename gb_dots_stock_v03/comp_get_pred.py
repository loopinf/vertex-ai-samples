from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)

def predict(
    ver : str,
    model01 : Input[Model],
    model02 : Input[Model],
    model03 : Input[Model],
    predict_dataset: Input[Dataset],
    daily_recom_dataset: Output[Dataset]
):
    import pandas as pd
    from catboost import CatBoostClassifier


    df_predict = pd.read_pickle(predict_dataset.path)
    print(f'shape of df_predict : {df_predict.shape}')

    cols_indicator = ['code', 'name','date']
    X_pred = df_predict.drop(columns=cols_indicator)
    X_indi = df_predict[cols_indicator].reset_index(drop=True)

    def get_pred_single(model):
        model_ = CatBoostClassifier()
        model_.load_model(model.path)
        pred = model_.predict(X_pred)
        pred = pd.DataFrame(pred, columns=['Prediction']).reset_index(drop=True)
      
        pred_prob = model_.predict_proba(X_pred)
        pred_prob = pd.DataFrame(pred_prob, columns=['Prob01','Prob02']).reset_index(drop=True)
       
        pred_each = pd.concat([X_indi, pred, pred_prob], axis=1)
        pred_each = pred_each[pred_each.Prediction > 0]
        print('a', pred_each)
        return pred_each
    
    df_pred01 = get_pred_single(model01)
    df_pred02 = get_pred_single(model02)
    df_pred03 = get_pred_single(model03)

    df_pred_all = pd.concat([df_pred01, df_pred02, df_pred03], axis=0)

    print('b', df_pred_all)
    df_pred_mean= df_pred_all.groupby(['name', 'code', 'date']).mean() # apply mean to duplicated recommends
    df_pred_mean = df_pred_mean.reset_index()
    df_pred_mean = df_pred_mean.sort_values(by='Prob02', ascending=False) # high probability first

    df_pred_mean.drop_duplicates(subset=['code', 'date'], inplace=True) 
    print('c', df_pred_mean)

    # # Load stored prediction result
    # try :        
    #     df_pred_stored = pd.read_pickle(f'/gcs/pipeline-dots-stock/bong_predictions/bong_{ver}.pkl')
    #     dates_in_stored = df_pred_stored.date.unique().tolist()
    # except :
    #     print('exception')
    #     df_pred_stored = pd.DataFrame()
    #     dates_in_stored = []
    

    # if df_pred_mean.date.iloc[0] not in dates_in_stored:
    #     df_pred_new = df_pred_stored.append(df_pred_mean)
    # else :
    #     df_pred_new = df_pred_stored

    # df_pred_new.to_pickle(daily_recom_dataset.path)
    df_pred_mean.to_pickle(daily_recom_dataset.path)
    # df_pred_new.to_pickle(f'/gcs/pipeline-dots-stock/bong_predictions/bong_{ver}.pkl')
    


