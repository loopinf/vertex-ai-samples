from typing import NamedTuple
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)

def conditioning_dataset(
  ml_dataset : Input[Dataset],
  pre_processed_dataset : Output[Dataset]
):

    import pandas as pd
    import FinanceDataReader as fdr
    import numpy as np

    df_ml_dataset = pd.read_pickle(ml_dataset.path)

    # Dataframe Copy 
    df_preP = df_ml_dataset.copy()

    # Re-arrange column names
    cols_ohlcv_x = ['open_x', 'high_x', 'low_x', 'close_x', 'volume_x']
    cols_ohlcv_y = ['open_y', 'high_y', 'low_y', 'close_y', 'volume_y', 'change_y']
    df_preP = df_preP.drop(columns=cols_ohlcv_x+cols_ohlcv_y)

    df_preP.rename(columns={"change_x" : "change"}, inplace=True)     

    # drop SPACs
    stock_names = pd.Series(df_preP.name.unique())
    stock_names_SPAC = stock_names[ stock_names.str.contains('스팩')].tolist()

    df_preP = df_preP.where( 
                lambda df : ~df.name.isin(stock_names_SPAC)
                ).dropna(subset=['name'])

    # # drop KODEX
    # stock_names = pd.Series(df_preP.name.unique())
    # stock_names_KODEX = stock_names[ stock_names.str.contains('KODEX')].tolist()

    # df_preP = df_preP.where( 
    #             lambda df : ~df.name.isin(stock_names_KODEX)
    #             ).dropna(subset=['name'])

    # # drop ETN
    # stock_names = pd.Series(df_preP.name.unique())
    # stock_names_ETN = stock_names[ stock_names.str.contains('ETN')].tolist()

    # df_preP = df_preP.where( 
    #             lambda df : ~df.name.isin(stock_names_ETN)
    #             ).dropna(subset=['name'])

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

    df_preP.to_pickle(pre_processed_dataset.path)