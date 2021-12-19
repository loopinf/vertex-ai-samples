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
from comps_calc_cos_similarity.comp_calc_df_snapshot import calc_df_snapshot
from comps_calc_cos_similarity.comp_calc_cos_similarity import calc_cos_similar
from comps_calc_cos_similarity.comp_calc_cos_similarity_occc import calc_cos_similar_occc
from comps_calc_cos_similarity.comp_eval_cos_simil import eval_cos_simil
from comps_update_hotstock.comp_market_watch import calc_market_watch
from comps_calc_cos_similarity.comp_add_price_on_pattern import add_price_on_pattern
from comps_calc_cos_similarity.comp_bigquery_sql import create_market_snap_top30_eval

comp_set_default = comp.create_component_from_func_v2(
                                            set_defaults,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            )
comp_update_df_markets = comp.create_component_from_func_v2(
                                            update_df_markets, 
                                            base_image="gcr.io/dots-stock/python-img-v5.2"
                                            )  
comp_calc_df_snapshot = comp.create_component_from_func_v2(
                                            calc_df_snapshot, 
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            packages_to_install=['google-cloud-bigquery==1.21.0'],
                                            )  
comp_calc_market_watch = comp.create_component_from_func_v2(
                                            calc_market_watch,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            packages_to_install=['google-cloud-bigquery==1.21.0'],
                                            )  
comp_calc_cos_similars = comp.create_component_from_func_v2(
                                            calc_cos_similar,
                                            base_image="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-9:latest",
                                            packages_to_install=['pandas_gbq']
                                            )           
comp_calc_cos_similars_occc = comp.create_component_from_func_v2(
                                            calc_cos_similar_occc,
                                            base_image="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-9:latest",
                                            packages_to_install=['pandas_gbq']
                                            )           
comp_add_price_on_pattern = comp.create_component_from_func_v2(
                                            add_price_on_pattern,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            packages_to_install=['google-cloud-bigquery==1.21.0'],
                                            )
comp_eval_cos_simil = comp.create_component_from_func_v2(
                                            eval_cos_simil,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            packages_to_install=['pandas_gbq']
                                            )           
comp_create_market_snap_top30_eval = comp.create_component_from_func_v2(
                                            create_market_snap_top30_eval,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            packages_to_install=['google-cloud-bigquery==1.21.0'],
)
# create pipeline 
#########################################
job_file_name='gb-pipeline-calc-cos-similars-and-evaluate-it.json'
@dsl.pipeline(
  name=job_file_name.split('.json')[0],
  pipeline_root=PIPELINE_ROOT,
)    
def create_awesome_pipeline():
  op_set_default = comp_set_default()

  # date_ref = '20211215'
  with dsl.Condition(op_set_default.outputs['isBusinessDay'] == 'yes'):

    op_get_df_markets = comp_update_df_markets(
        date_ref = op_set_default.outputs['date_ref']
    )
    op_calc_df_snapshot = comp_calc_df_snapshot(
        date_ref = op_set_default.outputs['date_ref']
    ).after(op_get_df_markets)
    op_calc_market_watch = comp_calc_market_watch(
        date_ref = op_set_default.outputs['date_ref']
    ).after(op_get_df_markets)

    op_calc_cos_similars_kernel3 = comp_calc_cos_similars(
        date_ref = op_set_default.outputs['date_ref'],
        # date_ref = date_ref,
        kernel_size = '3',
        comp_result = op_get_df_markets.output
    )
    op_calc_cos_similars_kernel6 = comp_calc_cos_similars(
        date_ref = op_set_default.outputs['date_ref'],
        # date_ref = date_ref,
        kernel_size = '6',
        comp_result = op_get_df_markets.output
    )
    op_calc_cos_similars_occc_10 = comp_calc_cos_similars_occc(
        date_ref = op_set_default.outputs['date_ref'],
        # date_ref = date_ref,
        kernel_size = '10',
        comp_result = op_get_df_markets.output
    )
    op_calc_cos_similars_occc_20 = comp_calc_cos_similars_occc(
        date_ref = op_set_default.outputs['date_ref'],
        # date_ref = date_ref,
        kernel_size = '20',
        comp_result = op_get_df_markets.output
    )

    op_add_price_on_pattern3 = comp_add_price_on_pattern(
        date_ref = op_set_default.outputs['date_ref'],
        kernel_size = '3'
        ).after(op_calc_cos_similars_kernel3)
    op_add_price_on_pattern6 = comp_add_price_on_pattern(
        date_ref = op_set_default.outputs['date_ref'],
        kernel_size = '6'
        ).after(op_calc_cos_similars_kernel6)
    op_add_price_on_pattern10 = comp_add_price_on_pattern(
        date_ref = op_set_default.outputs['date_ref'],
        kernel_size = '10'
        ).after(op_calc_cos_similars_occc_10)
    op_add_price_on_pattern20 = comp_add_price_on_pattern(
        date_ref = op_set_default.outputs['date_ref'],
        kernel_size = '20'
        ).after(op_calc_cos_similars_occc_20)

    op_eval_cos_simil = comp_eval_cos_simil(
        date_ref = op_set_default.outputs['date_ref'],
        # date_ref = date_ref,
    ).after(op_calc_cos_similars_kernel3, 
            op_calc_cos_similars_kernel6, 
            op_calc_cos_similars_occc_10, 
            op_calc_cos_similars_occc_20)

    op_create_market_snap_top30_eval = comp_create_market_snap_top30_eval(
        date_ref = op_set_default.outputs['date_ref'],
    ).after(op_eval_cos_simil)

    experimental.run_as_aiplatform_custom_job(
      op_calc_cos_similars_kernel3, machine_type='n1-standard-8', accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count="1"
    )
    experimental.run_as_aiplatform_custom_job(
      op_calc_cos_similars_kernel6, machine_type='n1-standard-8', accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count="1"
    )
    experimental.run_as_aiplatform_custom_job(
      op_calc_cos_similars_occc_10, machine_type='n1-standard-8', accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count="1"
    )
    experimental.run_as_aiplatform_custom_job(
      op_calc_cos_similars_occc_20, machine_type='n1-standard-8', accelerator_type="NVIDIA_TESLA_T4",
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

# when you want to run this script imediately, use it will create a pipeline
response = api_client.create_run_from_job_spec(
  job_spec_path=job_file_name,
  enable_caching= True,
  pipeline_root=PIPELINE_ROOT,
)

# # when you want to run this script on schedule, use it will create a pipeline
# response = api_client.create_schedule_from_job_spec(
#     job_spec_path=job_file_name,
#     schedule="58 15 * * 1-5",
#     time_zone="Asia/Seoul",
#     enable_caching = False,
# )