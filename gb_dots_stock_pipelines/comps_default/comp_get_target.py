from kfp.components import InputPath, OutputPath
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)

def get_target(
  # full_adj_prices_dataset_path: InputPath('DataFrame'),
  # target_dataset_path: OutputPath('DataFrame')
  full_adj_prices_dataset: Input[Dataset],
  target_dataset: Output[Dataset]
):
  import pandas as pd
  import numpy as np

  def make_target(df):

    df_ = df.copy()

    df_.sort_values(by='date', inplace=True)
    df_['high_p1'] = df_.high.shift(-1)
    df_['high_p2'] = df_.high.shift(-2)
    df_['high_p3'] = df_.high.shift(-3)

    df_['close_p1'] = df_.close.shift(-1)
    df_['close_p2'] = df_.close.shift(-2)
    df_['close_p3'] = df_.close.shift(-3)

    df_['low_p1'] = df_.low.shift(-1)
    df_['low_p2'] = df_.low.shift(-2)
    df_['low_p3'] = df_.low.shift(-3)

    df_['change_p1'] = (df_.close_p1 - df_.close) / df_.close
    df_['change_p2'] = (df_.close_p2 - df_.close) / df_.close
    df_['change_p3'] = (df_.close_p3 - df_.close) / df_.close

    df_['change_low_p1'] = (df_.low_p1 - df_.close) / df_.close
    df_['change_low_p2'] = (df_.low_p2 - df_.close) / df_.close
    df_['change_low_p3'] = (df_.low_p3 - df_.close) / df_.close

    df_['change_p1_over5'] = df_['change_p1'] > 0.05
    df_['change_p2_over5'] = df_['change_p2'] > 0.05
    df_['change_p3_over5'] = df_['change_p3'] > 0.05

    df_['change_low_p1_over10'] = df_['change_low_p1'] > 0.1
    df_['change_low_p2_over10'] = df_['change_low_p2'] > 0.1
    df_['change_low_p3_over10'] = df_['change_low_p3'] > 0.1

    df_['change_p1_over1'] = df_['change_p1'] > 0.01

    df_['change_p1_over10'] = df_['change_p1'] > 0.1
    df_['change_p2_over10'] = df_['change_p2'] > 0.1
    df_['change_p3_over10'] = df_['change_p3'] > 0.1

    df_['close_high_1'] = (df_.high_p1 - df_.close) / df_.close
    df_['close_high_2'] = (df_.high_p2 - df_.close) / df_.close
    df_['close_high_3'] = (df_.high_p3 - df_.close) / df_.close

    df_['close_high_1_over10'] = df_['close_high_1'] > 0.1
    df_['close_high_2_over10'] = df_['close_high_2'] > 0.1
    df_['close_high_3_over10'] = df_['close_high_3'] > 0.1

    df_['close_high_1_over5'] = df_['close_high_1'] > 0.05
    df_['close_high_2_over5'] = df_['close_high_2'] > 0.05
    df_['close_high_3_over5'] = df_['close_high_3'] > 0.05
    
    df_['target_over10'] = np.logical_or.reduce([
                                  df_.close_high_1_over10,
                                  df_.close_high_2_over10,
                                  df_.close_high_3_over10])
    
    df_['target_over5'] = np.logical_or.reduce([
                                  df_.close_high_1_over5,
                                  df_.close_high_2_over5,
                                  df_.close_high_3_over5])
    
    df_['target_close_over_10'] = np.logical_or.reduce([
                                  df_.change_p1_over10,
                                  df_.change_p2_over10,
                                  df_.change_p3_over10])  
    
    df_['target_close_over_5'] = np.logical_or.reduce([
                                  df_.change_p1_over5,
                                  df_.change_p2_over5,
                                  df_.change_p3_over5]) 
    
    df_['target_low_over_10'] = np.logical_or.reduce([
                                  df_.change_low_p1_over10,
                                  df_.change_low_p2_over10,
                                  df_.change_low_p3_over10])

                                  
    df_['target_mclass_close_over10_under5'] = \
        np.where(df_['change_p1'] > 0.1, 
                1,  np.where(df_['change_p1'] > -0.05, 0, -1))                               

    df_['target_mclass_close_p2_over10_under5'] = \
        np.where(df_['change_p2'] > 0.1, 
                1,  np.where(df_['change_p2'] > -0.05, 0, -1))                               
                
    df_['target_mclass_close_p3_over10_under5'] = \
        np.where(df_['change_p3'] > 0.1, 
                1,  np.where(df_['change_p3'] > -0.05, 0, -1))                               
    df_.dropna(subset=['high_p3'], inplace=True)                               
    return df_

  def get_target_df(df_price):
    df_price.reset_index(inplace=True)
    df_price.columns = df_price.columns.str.lower()
    df_target = df_price.groupby('code').apply(lambda df: make_target(df))
    df_target = df_target.reset_index(drop=True)
    # df_target['date'] = df_target.date.str.replace('-', '')
    return df_target

  df_price = pd.read_pickle(full_adj_prices_dataset.path)
  print('df cols =>', df_price.columns)

  df_target = get_target_df(df_price=df_price)
  df_target.to_pickle(target_dataset.path)
