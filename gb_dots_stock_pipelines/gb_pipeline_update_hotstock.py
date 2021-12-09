# -*- coding: utf-8 -*-
import sys
import os

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

from comps_default.comp_set_defaults_v2 import set_defaults
from comps_update_hotstock.comp_update_hot_stocks import update_hot_stocks

comp_get_update_hot_stocks = comp.create_component_from_func(
                          update_hot_stocks,
                          base_image="gcr.io/dots-stock/python-img-v5.2", 
                          packages_to_install=['pandas_gbq']
)

job_file_name='gb-pipeline-update-hot-stocks-20211205.json'
@dsl.pipeline(
  name=job_file_name.split('.json')[0],
  pipeline_root=PIPELINE_ROOT
)    
def create_awesome_pipeline():

  op_update_price_daily = comp_get_update_hot_stocks()

compiler.Compiler().compile(
  pipeline_func=create_awesome_pipeline,
  package_path=job_file_name
)

api_client = AIPlatformClient(
    project_id=PROJECT_ID,
    region=REGION,
)

# response = api_client.create_run_from_job_spec(
#   job_spec_path=job_file_name,
#   enable_caching= False,
#   pipeline_root=PIPELINE_ROOT
# )

response = api_client.create_schedule_from_job_spec(
  job_spec_path=job_file_name,
  schedule="30 8-22 * * *",
  time_zone="Asia/Seoul",
  enable_caching = False,
)