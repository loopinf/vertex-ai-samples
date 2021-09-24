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

from comps_default.comp_set_defaults_v2 import set_defaults
from comps_update_price_daily.comp_update_price_daily import update_price_daily

comp_set_default = comp.create_component_from_func(
                          set_defaults,
                          base_image="gcr.io/dots-stock/python-img-v5.2",
)

comp_get_update_price_daily = comp.create_component_from_func(
                          update_price_daily,
                          base_image="gcr.io/dots-stock/python-img-v5.2",
)



job_file_name='gb-pipeline-update-price-daily-20210909.json'
@dsl.pipeline(
  name=job_file_name.split('.json')[0],
  pipeline_root=PIPELINE_ROOT
)    
def create_awesome_pipeline():
  op_set_default = comp_set_default()

  with dsl.Condition(op_set_default.outputs['isBusinessDay'] == 'yes'):
    op_update_price_daily = comp_get_update_price_daily()

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
  schedule="30 9-15 * * 1-5",
  time_zone="Asia/Seoul",
  enable_caching = False,
)

response = api_client.create_schedule_from_job_spec(
  job_spec_path=job_file_name,
  schedule="0 10-15 * * 1-5",
  time_zone="Asia/Seoul",
  enable_caching = False,
)

response = api_client.create_schedule_from_job_spec(
  job_spec_path=job_file_name,
  schedule="15 10-15 * * 1-5",
  time_zone="Asia/Seoul",
  enable_caching = False,
)

response = api_client.create_schedule_from_job_spec(
  job_spec_path=job_file_name,
  schedule="45 10-14 * * 1-5",
  time_zone="Asia/Seoul",
  enable_caching = False,
)