from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)
from typing import NamedTuple

def calc_returns(
    price_n_return_updated_dataset : Input[Dataset],
    # return_updated_dataset : Output[Dataset]
) -> NamedTuple(
    'Outputs', [
    ('Total_Returns', float),
    ('Max_returns' , float),
    ('Min_returns', float),
    ('Testing_Period', int),
    ('Days_return_is', int)
]):

    import pandas as pd

    df_return_updated = pd.read_pickle(price_n_return_updated_dataset.path)

    daily_return = []
    def calc_daily_return(df):

        df_ = df.sort_values(by='Prediction', ascending=False)
        df_ = df.head(10) 
              
        fr = df_.r1.mean()
        daily_return.append(fr)
        
    df_return_updated.groupby('date').apply(lambda df : calc_daily_return(df))

    #%%
    # 11 Calc sum
    total_return = float(sum(daily_return))
    max_r = float(max(daily_return))
    min_r = float(min(daily_return))
    testing_period = int(df_return_updated.date.unique().tolist().__len__())
    num_of_p_day = int(daily_return.__len__())
    

    

    return (total_return, max_r, min_r, testing_period, num_of_p_day)