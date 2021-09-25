from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)

def add_prices_n_returns(
    prediction_result : Input[Dataset],
    price_n_return_updated_dataset : Output[Dataset]
):

    import pandas as pd
    import FinanceDataReader as fdr

    df_pred_all = pd.read_pickle(prediction_result.path)

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

        df_['l_1'] = df_.low.shift(-1)
        df_['l_2'] = df_.low.shift(-2)
        df_['l_3'] = df_.low.shift(-3)

        return df_

    df_price_updated  = df_price.groupby('code').apply(lambda df: get_price_tracked(df))
    df_price_updated = df_price_updated[['date', 'code', 'c_1', 'c_2', 'c_3', 'l_1', 'l_2', 'l_3', 'close']]
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

        lr1 = (df.l_1 / df.close - 1) * 100
        lr1 = format(lr1, '.1f')

        lr2 = (df.l_2 / df.close - 1) * 100
        lr2 = format(lr2, '.1f')

        lr3 = (df.l_3 / df.close - 1) * 100
        lr3 = format(lr3, '.1f')

        df['r1'] = float(r1)
        df['r2'] = float(r2)
        df['r3'] = float(r3)

        df['lr1'] = float(lr1)
        df['lr2'] = float(lr2)
        df['lr3'] = float(lr3)

        return df

    df_return_updated = df_price_updated.apply(lambda row: calc_return(row), axis=1)

    df_return_updated.to_pickle(price_n_return_updated_dataset.path)