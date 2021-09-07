from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)

def predict(
    model01 : Input[Model],
    model02 : Input[Model],
    model03 : Input[Model],
    predict_dataset: Input[Dataset],
    daily_recom_dataset: Output[Dataset]
):
    import pandas as pd
    from catboost import CatBoostClassifier

    df_predict = pd.read_pickle(predict_dataset.path)
    cols_indicator = ['code', 'name','date']
    X_pred = df_predict.drop(columns=cols_indicator)
    X_indi = df_predict[cols_indicator]

    def get_pred_single(model):
        model = CatBoostClassifier()
        model.load_model(model.path)
        pred = model.predict(X_pred)
        pred = pd.DataFrame(pred, columns=['Prediction']).reset_index(drop=True)
        pred_prob = model.predict_proba(X_pred)
        pred_prob = pd.DataFrame(pred_prob, columns=['Prob01','Prob02']).reset_index(drop=True)
        return pd.concat([X_indi, pred, pred_prob], axis=1)
    
    pred01 = get_pred_single(model01)
    pred02 = get_pred_single(model02)
    pred03 = get_pred_single(model03)

