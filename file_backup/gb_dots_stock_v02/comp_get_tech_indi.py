from kfp.components import InputPath, OutputPath

def get_tech_indi(
  # date_ref: str,
  adj_price_dataset_path: InputPath('DataFrame'),
  techini_dataset_path: OutputPath('DataFrame'),
  
):
  from stockstats import StockDataFrame as Sdf
  # from sklearn.preprocessing import MaxAbsScaler
  from sklearn.preprocessing import maxabs_scale
  import pandas as pd
  import pickle
  class FeatureEngineer:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            user user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """
    TECHNICAL_INDICATORS_LIST = ['macd',
      'boll_ub',
      'boll_lb',
      'rsi_30',
      'dx_30',
      'close_30_sma',
      'close_60_sma',
      # 'mfi',
      ]

    # PERIOD_MAX = 60,

    def __init__(
      self,
      use_technical_indicator=True,
      tech_indicator_list=TECHNICAL_INDICATORS_LIST,
      user_defined_feature=False,
  ):
      self.use_technical_indicator = use_technical_indicator
      self.tech_indicator_list = tech_indicator_list
      self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df):
      """main method to do the feature engineering
      @:param config: source dataframe
      @:return: a DataMatrices object
      """
      #clean data
      # df = self.clean_data(df)
      
      # add technical indicators using stockstats
      if self.use_technical_indicator == True:
        df = self.add_technical_indicator(df)
        print("Successfully added technical indicators")

      # add user defined feature
      if self.user_defined_feature == True:
        df = self.add_user_defined_feature(df)
        print("Successfully added user defined features")

      # fill the missing values at the beginning and the end
      df = df.fillna(method="bfill").fillna(method="ffill")
      return df
    
    def clean_data(self, data):
      """
      clean the raw data
      deal with missing values
      reasons: stocks could be delisted, not incorporated at the time step 
      :param data: (df) pandas dataframe
      :return: (df) pandas dataframe
      """
      df = data.copy()
      df=df.sort_values(['date','tic'],ignore_index=True) ##
      df.index = df.date.factorize()[0]
      merged_closes = df.pivot_table(index = 'date',columns = 'tic', values = 'close')
      merged_closes = merged_closes.dropna(axis=1)
      tics = merged_closes.columns
      df = df[df.tic.isin(tics)]
      return df

    def add_technical_indicator(self, data):
      """
      calculate technical indicators
      use stockstats package to add technical inidactors
      :param data: (df) pandas dataframe
      :return: (df) pandas dataframe
      """
      df = data.copy()
      df = df.sort_values(by=['tic','date'])
      stock = Sdf.retype(df.copy())
      unique_ticker = stock.tic.unique()

      for indicator in self.tech_indicator_list:
        indicator_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
          try:
            temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
            temp_indicator = pd.DataFrame(temp_indicator)
            temp_indicator['tic'] = unique_ticker[i]
            temp_indicator['date'] = df[df.tic == unique_ticker[i]]['date'].to_list()
            indicator_df = indicator_df.append(
                temp_indicator, ignore_index=True
            )
          except Exception as e:
            print(e)
        df = df.merge(indicator_df[['tic','date',indicator]],on=['tic','date'],how='left')
      df = df.sort_values(by=['date','tic'])
      return df

    def add_user_defined_feature(self, data):
        """
        add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=['tic','date'])
        df["daily_return"] = df.groupby('tic').close.pct_change(1)
        df['return_lag_1']=df.groupby('tic').close.pct_change(2)
        df['return_lag_2']=df.groupby('tic').close.pct_change(3)
        df['return_lag_3']=df.groupby('tic').close.pct_change(4)
        
        # bollinger band - relative
        df['bb_u_ratio'] = df.boll_ub / df.close # without groupby
        df['bb_l_ratio'] = df.boll_lb / df.close # don't need groupby

        # oh oc ol ratio 
        df['oh_ratio'] = (df.high - df.open) / df.open
        df['oc_ratio'] = (df.close - df.open) / df.open
        df['ol_ratio'] = (df.low - df.open) / df.open

        df['ch_ratio'] = (df.high - df.close) / df.close

        # macd - relative
        df['max_scale_MACD'] = df.groupby('tic').macd.transform(
            lambda x: maxabs_scale(x))

        # custom volume indicator
        def volume_change_wrt_10_max(df):
          return df.volume / df.volume.rolling(10).max()
        def volume_change_wrt_5_max(df):
          return df.volume / df.volume.rolling(5).max()
        def volume_change_wrt_20_max(df):
          return df.volume / df.volume.rolling(20).max()

        def volume_change_wrt_10_mean(df):
          return df.volume / df.volume.rolling(10).mean()
        def volume_change_wrt_5_mean(df):
          return df.volume / df.volume.rolling(5).mean()
        def volume_change_wrt_20_mean(df):
          return df.volume / df.volume.rolling(20).mean()

        df['volume_change_wrt_10max'] = (
            df.groupby('tic')
            .apply(lambda df: volume_change_wrt_10_max(df))
            .reset_index(drop=True)
            )
        df['volume_change_wrt_5max'] = (
            df.groupby('tic')
            .apply(lambda df: volume_change_wrt_5_max(df))
            .reset_index(drop=True)
            )
        df['volume_change_wrt_20max'] = (
            df.groupby('tic')
            .apply(lambda df: volume_change_wrt_20_max(df))
            .reset_index(drop=True)
            )


        df['volume_change_wrt_10mean'] = (
            df.groupby('tic')
            .apply(lambda df: volume_change_wrt_10_mean(df))
            .reset_index(drop=True)
            )
        df['volume_change_wrt_5mean'] = (
            df.groupby('tic')
            .apply(lambda df: volume_change_wrt_5_mean(df))
            .reset_index(drop=True)
            )
        df['volume_change_wrt_20mean'] = (
            df.groupby('tic')
            .apply(lambda df: volume_change_wrt_20_mean(df))
            .reset_index(drop=True)
            )

        

        # close ratio rolling min max
        def close_ratio_wrt_10_max(df):
          return df.close / df.close.rolling(10).max() 
        def close_ratio_wrt_10_min(df):
          return df.close / df.close.rolling(10).min() 

        df['close_ratio_wrt_10max'] = (
          df.groupby('tic')
          .apply(lambda df: close_ratio_wrt_10_max(df))
          .reset_index(drop=True)
        )
        df['close_ratio_wrt_10min'] = (
          df.groupby('tic')
          .apply(lambda df: close_ratio_wrt_10_min(df))
          .reset_index(drop=True)
        )
        
        return df
  
  df_price = pd.read_pickle(adj_price_dataset_path)
  
  print('size =>', df_price.shape)
  print('cols =>', df_price.columns)

  df_price.columns = df_price.columns.str.lower()
  df_price.rename(columns={'code':'tic'}, inplace=True)
  fe = FeatureEngineer(user_defined_feature=True)
  df_process = fe.preprocess_data(df_price)
  df_process.rename(columns={'tic':'code'}, inplace=True)

  df_process.to_pickle(techini_dataset_path)