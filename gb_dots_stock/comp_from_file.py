# -*- coding: utf-8 -*-
import sys
import os

from pandas.io import pickle
# import pandas as pd

PROJECT_ID = "dots-stock"  # @param {type:"string"}
REGION = "us-central1"  # @param {type:"string"}
USER = "shkim01"  # <---CHANGE THIS
BUCKET_NAME = "gs://pipeline-dots-stock"  # @param {type:"string"}
PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root/{USER}"

from typing import NamedTuple

from kfp import dsl
from kfp.v2 import compiler
import kfp.components as comp
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics,
                        component)
from kfp.v2.google.client import AIPlatformClient

from test import test

print_op = comp.create_component_from_func(
                                            test,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            )
#########################################
# create pipeline #######################
#########################################
job_file_name='test.json'
@dsl.pipeline(
  name=job_file_name.split('.json')[0],
  pipeline_root=PIPELINE_ROOT
)    
def create_awesome_pipeline():
  op = print_op()

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





######################



# @component(
#     base_image="amancevice/pandas:1.3.2-slim"
# )
# def get_univ_for_price(
#   # date_ref: str,
#   base_item_dataset: Input[Dataset],
#   bros_dataset: Input[Dataset],
#   univ_dataset: Output[Dataset],
# ):
#   import pandas as pd
#   import logging
#   import json
#   logger = logging.getLogger(__name__)
#   FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
#   logging.basicConfig(format=FORMAT)
#   logger.setLevel(logging.DEBUG)

#   # base item
#   df_top30s = pd.read_csv(base_item_dataset.path, 
#                        index_col=0, 
#                        dtype={'날짜': str}).reset_index(drop=True)

#   # load edge_list to make bros
#   df_ed = pd.read_csv(bros_dataset.path, index_col=0).reset_index(drop=True)
#   df_ed_r = df_ed.copy() 
#   df_ed_r.rename(columns={'target':'source', 'source':'target'}, inplace=True)
#   df_ed2 = df_ed.append(df_ed_r, ignore_index=True)
#   df_ed2['date'] = pd.to_datetime(df_ed2.date).dt.strftime('%Y%m%d')

#   dic_univ = {}
#   for date, df in df_top30s.groupby('날짜'):
#     logger.debug(f'date: {date}')
#     l_top30 = df.종목코드.to_list()
#     l_bro = df_ed2[(df_ed2.date == date) & 
#                   (df_ed2.source.isin(l_top30))].target.unique().tolist()

#     dic_univ[date] = list(set(l_top30 + l_bro ))

#   with open(univ_dataset.path, 'w', encoding='utf8') as f:
#     json.dump(dic_univ, f)
