# -*- coding: utf-8 -*-
import sys
import os
# import pandas as pd

PROJECT_ID = "dots-stock"  # @param {type:"string"}
REGION = "us-central1"  # @param {type:"string"}
USER = "shkim01"  # <---CHANGE THIS
BUCKET_NAME = "gs://pipeline-dots-stock"  # @param {type:"string"}
PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root/{USER}"

from typing import NamedTuple

from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics,
                        component)
from kfp.v2.google.client import AIPlatformClient

@component(
  base_image="gcr.io/dots-stock/python-img-v5.2",
  # packages_to_install=['catboost', 'scikit-learn', 'ipywidgets']
)
def update_price(
  predictions : Input[Dataset]
):

    import pandas as pd
    import FinanceDataReader as fdr
    import pickle
    import os


    df_pred_result = pd.read_pickle(predictions.path)

    def get_price_adj(code, start):
        return fdr.DataReader(code, start=start)    

    def get_price(codes, date_start):

        df_price = pd.DataFrame()
        for code in codes :      
            df_ = get_price_adj(code, date_start)
            df_['code'] = code
            df_price = df_price.append(df_)

            # print(df_price.shape, code)      
        return df_price

    def get_price_tracked(df):

        df_ = df.copy()
        df_.sort_values(by='date', inplace=True)
        df_['c_1'] = df_.close.shift(-1)
        df_['c_2'] = df_.close.shift(-2)
        df_['c_3'] = df_.close.shift(-3)

        return df_

    def get_up_to_date(df):

        l_dates = df.date.unique().tolist()

        dates_to_update = l_dates[-4:]

        df_to_hold = df[~df.date.isin(dates_to_update)]
        df_to_update = df[df.date.isin(dates_to_update)]

        codes_to_update = df_to_update.code.unique().tolist()

        date_start = dates_to_update[0]
        df_price = get_price(codes_to_update, date_start)

        df_price.reset_index(inplace=True)
        df_price.columns = df_price.columns.str.lower()
        df_price['date'] = df_price.date.dt.strftime('%Y%m%d')

        df_price_updated  = df_price.groupby('code').apply(lambda df: get_price_tracked(df))
        df_price_updated = df_price_updated.reset_index(drop=True)

        # try :
        #     df_to_update.drop(columns=['c_1', 'c_2', 'c_3'], inplace=True)
        # except :
        #     pass

        df_to_update = df_to_update.merge(
                                df_price_updated,
                                left_on=['date', 'code'],
                                right_on=['date', 'code'] )
        df_to_update.fillna(0, inplace=True)

        df_updated = df_to_hold.append(df_to_update)

        return df_updated

    df_pred_result_upd = get_up_to_date(df_pred_result)


    dir = "/gcs/pipeline-dots-stock/result_bong"  
    file_name = 'bong01.pkl'

    path = os.path.join(dir, file_name)

    with open(path, 'wb') as f:
        pickle.dump(df_pred_result_upd, f)

#########################################
# create pipeline #######################
#########################################
job_file_name='price-update-20210905.json'
@dsl.pipeline(
  name=job_file_name.split('.json')[0],
  pipeline_root=PIPELINE_ROOT
)    
def create_awesome_pipeline():
  op_set_defaults = update_price()


compiler.Compiler().compile(
  pipeline_func=create_awesome_pipeline,
  package_path=job_file_name
)

api_client = AIPlatformClient(
    project_id=PROJECT_ID,
    region=REGION,
)

response = api_client.create_run_from_job_spec(
  job_spec_path=job_file_name,
  enable_caching= True,
  pipeline_root=PIPELINE_ROOT
)