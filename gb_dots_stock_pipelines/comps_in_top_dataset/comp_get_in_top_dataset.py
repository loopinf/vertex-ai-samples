from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)
from kfp.components import InputPath, OutputPath

def get_in_top_dataset(
    top_n : int,  
    market_info_dataset: Input[Dataset] ,
    adj_price_dataset: Input[Dataset],
    in_top_dataset : Output[Dataset]
    ):

    import FinanceDataReader as fdr
    import pandas as pd
    from multiprocessing import Pool

    df_market = pd.read_pickle(market_info_dataset.path)
    df_adj_price = pd.read_pickle(adj_price_dataset.path)

    df_in_top = df_market.merge(
                                df_adj_price,
                                left_on = ['date', 'code'],
                                right_on = ['date', 'code']                          
                            )
    # print(df_in_top.head())

    df_in_top['in_top_30'] = df_in_top.Rank < 30
    df_in_top['in_top_50'] = df_in_top.Rank < 50
    df_in_top['in_top_70'] = df_in_top.Rank < 70


    df_in_top['in_top_30'] = df_in_top.in_top_30.astype(int)
    df_in_top['in_top_50'] = df_in_top.in_top_50.astype(int)
    df_in_top['in_top_70'] = df_in_top.in_top_70.astype(int)
    
    df_in_top['in_top_sig_30'] = df_in_top['in_top_30'] * df_in_top['change']
    df_in_top['in_top_sig_50'] = df_in_top['in_top_50'] * df_in_top['change']
    df_in_top['in_top_sig_70'] = df_in_top['in_top_70'] * df_in_top['change']

    df_in_top.to_pickle(in_top_dataset.path)