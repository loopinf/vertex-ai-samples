from kfp.components import InputPath, OutputPath
from kfp.v2.dsl import (Dataset,
                        Input,
                        Output
)

def get_features(
  market_info_dataset: Input[Dataset],
  bros_univ_dataset: Input[Dataset],
  features_dataset: Output[Dataset]
  ):
  
  import pandas as pd
  import numpy as np
  from collections import Counter

  df_market = pd.read_pickle(market_info_dataset.path)

  # 등락률 -1 
  df_market = df_market.sort_values('날짜')
  df_market['return_-1'] = df_market.groupby('종목코드').등락률.shift(1)

  #df_ed 가져오기
  df_ed = pd.read_pickle(bros_univ_dataset.path)

  df_ed_r = df_ed.copy() 
  df_ed_r.rename(columns={'target':'source', 'source':'target'}, inplace=True)
  df_ed2 = df_ed.append(df_ed_r, ignore_index=True)
  df_ed2['date'] = pd.to_datetime(df_ed2.date).dt.strftime('%Y%m%d')

  cols = ['종목코드', '종목명', '날짜', '순위_상승률', '시가총액']
  df_mkt_ = df_market[cols]

  cols_market = [ '종목코드','날짜','등락률','return_-1']
  cols_bro = ['source','target','period','date']

  # merge
  df_ed2_1 = (df_ed2[cols_bro]
                  .merge(df_market[cols_market], 
                      left_on=['target','date'],
                      right_on=['종목코드','날짜'])
                  .rename(columns={'등락률':'target_return',
                  'return_-1':'target_return_-1'}))
  df_ed2_1 = df_ed2_1[['source', 'target', 'period', 'date', 
                      'target_return', 'target_return_-1']]
  
  df_tmp = df_mkt_.merge(df_ed2_1, 
          left_on=['날짜','종목코드'], 
          right_on=['date', 'source'], 
          suffixes=('','_x'),
          how='left')
  df_tmp.drop(columns=['종목코드','날짜'], inplace=True)
  df_tmp.dropna(subset=['target'], inplace=True)

  def get_upbro_ratio(df):
      '''df : '''
      return (
            sum(df.target_return > 0) / df.shape[0], # 그날 상승한 친구들의 비율
              df.shape[0], # 그날 친구들 수
              df.target_return.mean(), # 그날 모든 친구들 상승률의 평균
              df[df.target_return > 0].target_return.mean(), # 그날 오른 친구들의 평균
              df['target_return_-1'].mean(),# 전날 친구들 평균상승률
              sum(df['target_return_-1'] > 0) / df.shape[0],# 전날 상승한 친구들 비율
              df[df['target_return_-1'] > 0]['target_return_-1'].mean(),# 전날 상승한 친구들 평균
              )

  bro_up_ratio = (df_tmp.groupby(['date','source','period'])
      .apply(lambda df: get_upbro_ratio(df))
      .reset_index()
      .rename(columns={0:'bro_up_ratio'})
      )
  
  bro_up_ratio[['bro_up_ratio','n_bros', 'all_bro_rtrn_mean', 'up_bro_rtrn_mean',
                  'all_bro_rtrn_mean_ystd', 'bro_up_ratio_ystd', 'up_bro_rtrn_mean_ystd']] = \
      pd.DataFrame(bro_up_ratio.bro_up_ratio.tolist(), index=bro_up_ratio.index) 
  

  df_tmp = df_tmp.merge(bro_up_ratio, on=['date','source','period'], how='left')
  df_tmp['up_bro_ratio_20'] = df_tmp[df_tmp.period == 20].bro_up_ratio
  df_tmp['up_bro_ratio_40'] = df_tmp[df_tmp.period == 40].bro_up_ratio
  df_tmp['up_bro_ratio_60'] = df_tmp[df_tmp.period == 60].bro_up_ratio
  df_tmp['up_bro_ratio_90'] = df_tmp[df_tmp.period == 90].bro_up_ratio
  df_tmp['up_bro_ratio_120'] = df_tmp[df_tmp.period == 120].bro_up_ratio

  df_tmp.fillna(0, inplace=True) #친구가 없는 종목의 bro_up_ratio를 0으로 만들기
  df_tmp.drop(columns=['bro_up_ratio'], inplace=True)

  df_tmp['n_bro_20'] = df_tmp[df_tmp.period == 20].n_bros
  df_tmp['n_bro_40'] = df_tmp[df_tmp.period == 40].n_bros
  df_tmp['n_bro_60'] = df_tmp[df_tmp.period == 60].n_bros
  df_tmp['n_bro_90'] = df_tmp[df_tmp.period == 90].n_bros
  df_tmp['n_bro_120'] = df_tmp[df_tmp.period == 120].n_bros

  df_tmp.fillna(0, inplace=True) #친구가 없는 종목의 n_bros를 0으로 만들기
  df_tmp.drop(columns=['n_bros'], inplace=True)

  df_tmp['all_bro_rtrn_mean_20'] = df_tmp[df_tmp.period == 20].all_bro_rtrn_mean
  df_tmp['all_bro_rtrn_mean_40'] = df_tmp[df_tmp.period == 40].all_bro_rtrn_mean
  df_tmp['all_bro_rtrn_mean_60'] = df_tmp[df_tmp.period == 60].all_bro_rtrn_mean
  df_tmp['all_bro_rtrn_mean_90'] = df_tmp[df_tmp.period == 90].all_bro_rtrn_mean
  df_tmp['all_bro_rtrn_mean_120'] = df_tmp[df_tmp.period == 120].all_bro_rtrn_mean

  df_tmp.fillna(0, inplace=True) #친구가 없는 종목의 n_bros를 0으로 만들기
  df_tmp.drop(columns=['all_bro_rtrn_mean'], inplace=True)

  df_tmp['up_bro_rtrn_mean_20'] = df_tmp[df_tmp.period == 20].up_bro_rtrn_mean
  df_tmp['up_bro_rtrn_mean_40'] = df_tmp[df_tmp.period == 40].up_bro_rtrn_mean
  df_tmp['up_bro_rtrn_mean_60'] = df_tmp[df_tmp.period == 60].up_bro_rtrn_mean
  df_tmp['up_bro_rtrn_mean_90'] = df_tmp[df_tmp.period == 90].up_bro_rtrn_mean
  df_tmp['up_bro_rtrn_mean_120'] = df_tmp[df_tmp.period == 120].up_bro_rtrn_mean

  df_tmp.fillna(0, inplace=True) #친구가 없는 종목의 n_bros를 0으로 만들기
  df_tmp.drop(columns=['up_bro_rtrn_mean'], inplace=True)

  df_tmp['all_bro_rtrn_mean_ystd_20'] = df_tmp[df_tmp.period == 20].all_bro_rtrn_mean_ystd
  df_tmp['all_bro_rtrn_mean_ystd_40'] = df_tmp[df_tmp.period == 40].all_bro_rtrn_mean_ystd
  df_tmp['all_bro_rtrn_mean_ystd_60'] = df_tmp[df_tmp.period == 60].all_bro_rtrn_mean_ystd
  df_tmp['all_bro_rtrn_mean_ystd_90'] = df_tmp[df_tmp.period == 90].all_bro_rtrn_mean_ystd
  df_tmp['all_bro_rtrn_mean_ystd_120'] = df_tmp[df_tmp.period == 120].all_bro_rtrn_mean_ystd

  df_tmp.fillna(0, inplace=True) #친구가 없는 종목의 n_bros를 0으로 만들기
  df_tmp.drop(columns=['all_bro_rtrn_mean_ystd'], inplace=True)

  df_tmp['bro_up_ratio_ystd_20'] = df_tmp[df_tmp.period == 20].bro_up_ratio_ystd
  df_tmp['bro_up_ratio_ystd_40'] = df_tmp[df_tmp.period == 40].bro_up_ratio_ystd
  df_tmp['bro_up_ratio_ystd_60'] = df_tmp[df_tmp.period == 60].bro_up_ratio_ystd
  df_tmp['bro_up_ratio_ystd_90'] = df_tmp[df_tmp.period == 90].bro_up_ratio_ystd
  df_tmp['bro_up_ratio_ystd_120'] = df_tmp[df_tmp.period == 120].bro_up_ratio_ystd

  df_tmp.fillna(0, inplace=True) #친구가 없는 종목의 n_bros를 0으로 만들기
  df_tmp.drop(columns=['bro_up_ratio_ystd'], inplace=True)

  df_tmp['up_bro_rtrn_mean_ystd_20'] = df_tmp[df_tmp.period == 20].up_bro_rtrn_mean_ystd
  df_tmp['up_bro_rtrn_mean_ystd_40'] = df_tmp[df_tmp.period == 40].up_bro_rtrn_mean_ystd
  df_tmp['up_bro_rtrn_mean_ystd_60'] = df_tmp[df_tmp.period == 60].up_bro_rtrn_mean_ystd
  df_tmp['up_bro_rtrn_mean_ystd_90'] = df_tmp[df_tmp.period == 90].up_bro_rtrn_mean_ystd
  df_tmp['up_bro_rtrn_mean_ystd_120'] = df_tmp[df_tmp.period == 120].up_bro_rtrn_mean_ystd

  df_tmp.fillna(0, inplace=True) #친구가 없는 종목의 n_bros를 0으로 만들기
  df_tmp.drop(columns=['up_bro_rtrn_mean_ystd'], inplace=True)

  # Features related with Rank

  df_rank = df_mkt_.copy()

  df_rank['in_top30'] = df_rank.순위_상승률 <= 30
  df_rank['rank_mean_10'] = df_rank.groupby('종목코드')['순위_상승률'].transform(
                              lambda x : x.rolling(10, min_periods=1).mean()
                          )

  df_rank['rank_mean_5'] = df_rank.groupby('종목코드')['순위_상승률'].transform(
                              lambda x : x.rolling(5, min_periods=1).mean()
                          )

  df_rank['in_top_30_5'] = df_rank.groupby('종목코드')['in_top30'].transform(
                              lambda x : x.rolling(5, min_periods=1).sum()
                          )

  df_rank['in_top_30_10'] = df_rank.groupby('종목코드')['in_top30'].transform(
                              lambda x : x.rolling(10, min_periods=1).sum()
                          )

  df_rank['in_top_30_20'] = df_rank.groupby('종목코드')['in_top30'].transform(
                              lambda x : x.rolling(20, min_periods=1).sum()
                          )

  # Merge DataFrames
  # cols_rank = ['종목코드', '날짜', 'in_top30', 'rank_mean_10', 'rank_mean_5', 'in_top_30_5', 'in_top_30_10']
  cols_tmp = ['source', 'date',
          'up_bro_ratio_20', 'up_bro_ratio_40',
        'up_bro_ratio_60', 'up_bro_ratio_90', 'up_bro_ratio_120', 'n_bro_20',
        'n_bro_40', 'n_bro_60', 'n_bro_90', 'n_bro_120', 'all_bro_rtrn_mean_20',
        'all_bro_rtrn_mean_40', 'all_bro_rtrn_mean_60', 'all_bro_rtrn_mean_90',
        'all_bro_rtrn_mean_120', 'up_bro_rtrn_mean_20', 'up_bro_rtrn_mean_40',
        'up_bro_rtrn_mean_60', 'up_bro_rtrn_mean_90', 'up_bro_rtrn_mean_120',
        'all_bro_rtrn_mean_ystd_20', 'all_bro_rtrn_mean_ystd_40',
        'all_bro_rtrn_mean_ystd_60', 'all_bro_rtrn_mean_ystd_90',
        'all_bro_rtrn_mean_ystd_120', 'bro_up_ratio_ystd_20',
        'bro_up_ratio_ystd_40', 'bro_up_ratio_ystd_60', 'bro_up_ratio_ystd_90',
        'bro_up_ratio_ystd_120', 'up_bro_rtrn_mean_ystd_20',
        'up_bro_rtrn_mean_ystd_40', 'up_bro_rtrn_mean_ystd_60',
        'up_bro_rtrn_mean_ystd_90', 'up_bro_rtrn_mean_ystd_120']

  df_feat_bro = df_tmp.drop_duplicates(subset=['source', 'date'])

  df_feats =df_rank.merge(df_feat_bro[cols_tmp],
                      left_on=['종목코드', '날짜'],
                      right_on=['source', 'date'],
                      how='left')

  # df_feats.fillna(0, inplace=True)
  df_feats.drop(columns=['source', 'date'], inplace=True)
  df_feats.rename(
            columns={
                '종목코드':'code',
                '종목명':'name',
                '순위_상승률':'rank',
                '날짜':'date',
                '시가총액':'mkt_cap',
                }, inplace=True)
                
  df_feats.fillna(0, inplace=True)
  
  df_feats.to_pickle(features_dataset.path)