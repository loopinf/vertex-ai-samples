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
from comps_calc_cos_similarity.comp_calc_cos_similarity_occc import calc_cos_similar_occc
from comps_calc_cos_similarity.comp_eval_cos_simil import eval_cos_simil


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
comp_calc_cos_similars_occc = comp.create_component_from_func_v2(
                                            calc_cos_similar_occc,
                                            base_image="asia-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-9:latest",
                                            packages_to_install=['pandas_gbq']
                                            )           

comp_eval_cos_simil = comp.create_component_from_func_v2(
                                            eval_cos_simil,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
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
  # op_set_default = comp_set_default()

  date_ref = '20211213'
  # with dsl.Condition(op_set_default.outputs['isBusinessDay'] == 'yes'):
  if True:

    # op_get_df_markets = comp_update_df_markets(
    #     date_ref = op_set_default.outputs['date_ref']
    # )

    # op_calc_cos_similars_kernel3 = comp_calc_cos_similars(
    #     # df_markets = op_get_df_markets.outputs['df_markets_update'],)
    #     date_ref = op_set_default.outputs['date_ref'],
    #     # date_ref = date_ref,
    #     kernel_size = '3',
    # )
    # op_calc_cos_similars_kernel6 = comp_calc_cos_similars(
    #     # df_markets = op_get_df_markets.outputs['df_markets_update'],)
    #     date_ref = op_set_default.outputs['date_ref'],
    #     # date_ref = date_ref,
    #     kernel_size = '6',
    # )
    # op_calc_cos_similars_occc_10 = comp_calc_cos_similars_occc(
    #     # df_markets = op_get_df_markets.outputs['df_markets_update'],)
    #     date_ref = op_set_default.outputs['date_ref'],
    #     # date_ref = date_ref,
    #     kernel_size = '10',
    # )
    # op_calc_cos_similars_occc_20 = comp_calc_cos_similars_occc(
    #     # df_markets = op_get_df_markets.outputs['df_markets_update'],)
    #     date_ref = op_set_default.outputs['date_ref'],
    #     # date_ref = date_ref,
    #     kernel_size = '20',
    # )

    op_eval_cos_simil = comp_eval_cos_simil(
        # date_ref = op_set_default.outputs['date_ref'],
        date_ref = date_ref,
        cal_cos_simil_1 = 'a',
        cal_cos_simil_2 = 'b',
        cal_cos_simil_3 = 'c', 
        cal_cos_simil_4 = 'd',
        # cal_cos_simil_1 = op_calc_cos_similars_kernel3.output,
        # cal_cos_simil_2 = op_calc_cos_similars_kernel6.output,
        # cal_cos_simil_3 = op_calc_cos_similars_occc_10.output,
        # cal_cos_simil_4 = op_calc_cos_similars_occc_20.output,
    )

    # experimental.run_as_aiplatform_custom_job(
    #   op_calc_cos_similars_kernel3, machine_type='n1-standard-8', accelerator_type="NVIDIA_TESLA_T4",
    #         accelerator_count="1"
    #  )
    # experimental.run_as_aiplatform_custom_job(
    #   op_calc_cos_similars_kernel6, machine_type='n1-standard-8', accelerator_type="NVIDIA_TESLA_T4",
    #         accelerator_count="1"
    #  )
    # experimental.run_as_aiplatform_custom_job(
    #   op_calc_cos_similars_occc_10, machine_type='n1-standard-8', accelerator_type="NVIDIA_TESLA_T4",
    #         accelerator_count="1"
    #  )
    # experimental.run_as_aiplatform_custom_job(
    #   op_calc_cos_similars_occc_20, machine_type='n1-standard-8', accelerator_type="NVIDIA_TESLA_T4",
    #         accelerator_count="1"
    #  )

compiler.Compiler().compile(
  pipeline_func=create_awesome_pipeline,
  package_path=job_file_name,
)

api_client = AIPlatformClient(
    project_id=PROJECT_ID,
    region=REGION,
)

# when you want to run this script imediately, use it will create a pipeline
response = api_client.create_run_from_job_spec(
  job_spec_path=job_file_name,
  enable_caching= True,
  pipeline_root=PIPELINE_ROOT,
)

# when you want to run this script on schedule, use it will create a pipeline
# response = api_client.create_schedule_from_job_spec(
#     job_spec_path=job_file_name,
#     schedule="30 16 * * 1-5",
#     time_zone="Asia/Seoul",
#     enable_caching = False,
# )