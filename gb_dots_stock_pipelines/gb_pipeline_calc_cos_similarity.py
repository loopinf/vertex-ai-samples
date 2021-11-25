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
from comps_calc_cos_similarity.comp_update_df_markets import update_df_markets
from comps_calc_cos_similarity.comp_calc_cos_similarity import calc_cos_similar

# TODO
# date 설정
# df_market update ( today, current -- latest )
# cosine similarity 계산


comp_set_default = comp.create_component_from_func_v2(
                                            set_defaults,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            )

comp_update_df_markets = comp.create_component_from_func_v2(
                                            update_df_markets, 
                                            base_image="gcr.io/dots-stock/python-img-v5.2"
                                            )  

comp_calc_cos_similars = comp.create_component_from_func_v2(
                                            calc_cos_similar,
                                            base_image="asia-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-9:latest",
                                            packages_to_install=['pandas_gbq']
                                            )           




# create pipeline 
#########################################
job_file_name='gb-pipeline-calc-cos-similars.json'
@dsl.pipeline(
  name=job_file_name.split('.json')[0],
  pipeline_root=PIPELINE_ROOT,
)    
def create_awesome_pipeline():

  op_set_default = comp_set_default()

  with dsl.Condition(op_set_default.outputs['isBusinessDay'] == 'yes'):

    op_get_df_markets = comp_update_df_markets(
        date_ref = op_set_default.outputs['date_ref'])

    op_calc_cos_similars = comp_calc_cos_similars(
        date_ref = op_set_default.outputs['date_ref'],
        df_markets = op_get_df_markets.outputs['df_markets_update'],)

    experimental.run_as_aiplatform_custom_job(
      op_calc_cos_similars, machine_type='n1-standard-4', accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count="1"
 )

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
  enable_caching= False,
  pipeline_root=PIPELINE_ROOT,
)

# response = api_client.create_schedule_from_job_spec(
#     job_spec_path=job_file_name,
#     schedule="30 16 * * 1-5",
#     time_zone="Asia/Seoul",
#     enable_caching = False,
# )