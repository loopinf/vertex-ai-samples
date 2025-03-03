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
from kfp.v2.google import experimental

from comps_default.comp_set_defaults_v2 import set_defaults
from comps_default.comp_get_market_info import get_market_info
from comps_default.comp_get_adj_price_daily import get_adj_prices_daily
from comps_calc_corr.comp_calc_corr import calc_corr_rolling5

comp_set_default = comp.create_component_from_func_v2(
                                            set_defaults,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            )

comp_get_market_info = comp.create_component_from_func_v2(
                                            get_market_info,
                                            base_image="gcr.io/dots-stock/python-img-v5.2"
                                            )

comp_get_adj_price_daily = comp.create_component_from_func_v2(
                                            get_adj_prices_daily,
                                            base_image="gcr.io/dots-stock/python-img-v5.2"
                                            )  

comp_calc_corr_rolling_5 = comp.create_component_from_func_v2(
                                            calc_corr_rolling5,
                                            base_image="gcr.io/dots-stock/python-img-v5.2"
                                            )           




# create pipeline 
#########################################
job_file_name='gb-pipeline-calc-corr-rolling-5-120days.json'
@dsl.pipeline(
  name=job_file_name.split('.json')[0],
  pipeline_root=PIPELINE_ROOT,
)    
def create_awesome_pipeline():

  op_set_default = comp_set_default()

  with dsl.Condition(op_set_default.outputs['isBusinessDay'] == 'yes'):

    op_get_market_info = comp_get_market_info(
        date_ref = op_set_default.outputs['date_ref'],
        n_days = 1 )

    op_get_adj_price_daily = comp_get_adj_price_daily(
        market_info_dataset = op_get_market_info.outputs['market_info_dataset'],
        date_ref = op_set_default.outputs['date_ref'])

    op_calc_corr_rolling5 = comp_calc_corr_rolling_5(
        date_ref = op_set_default.outputs['date_ref'],
        adj_price_dataset = op_get_adj_price_daily.outputs['adj_price_dataset'],)

    experimental.run_as_aiplatform_custom_job(
      op_calc_corr_rolling5, machine_type='e2-standard-4')


   
compiler.Compiler().compile(
  pipeline_func=create_awesome_pipeline,
  package_path=job_file_name,
)

api_client = AIPlatformClient(
    project_id=PROJECT_ID,
    region=REGION,
)

response = api_client.create_run_from_job_spec(
  job_spec_path=job_file_name,
  enable_caching= True,
  pipeline_root=PIPELINE_ROOT,
)

# response = api_client.create_schedule_from_job_spec(
#     job_spec_path=job_file_name,
#     schedule="30 16 * * 1-5",
#     time_zone="Asia/Seoul",
#     enable_caching = False,
# )