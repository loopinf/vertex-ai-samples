# Writer : S H Kim
# Date : 2021. 09. 07.

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

from comp_set_defaults_v2 import set_defaults
from comp_get_market_info import get_market_info
from comp_get_bros import get_bros
from comp_get_adj_price import get_adj_prices
from comp_get_full_adj_prices import get_full_adj_prices
from comp_get_features import get_features
from comp_get_target import get_target
from comp_get_tech_indi import get_tech_indi
from comp_get_full_tech_indi import get_full_tech_indi
from comp_get_ml_dataset import get_ml_dataset

from comp_prediction_daily_04 import get_prediction_04

from comp_update_price_daily import update_price_daily

comp_set_default = comp.create_component_from_func(
                                            set_defaults,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            )

comp_get_market_info = comp.create_component_from_func(
                                            get_market_info,
                                            base_image="gcr.io/dots-stock/python-img-v5.2"
                                            )

comp_get_bros = comp.create_component_from_func(
                                            get_bros,
                                            base_image="gcr.io/dots-stock/python-img-v5.2"
                                            )
                                            
comp_get_adj_price = comp.create_component_from_func(
                                            get_adj_prices,
                                            base_image="gcr.io/dots-stock/python-img-v5.2"
                                            )

comp_get_full_adj_price = comp.create_component_from_func(
                                            get_full_adj_prices,
                                            base_image="gcr.io/dots-stock/python-img-v5.2"
                                            )                       

comp_get_features = comp.create_component_from_func(
                                            get_features,
                                            base_image="gcr.io/dots-stock/python-img-v5.2"
                                            )

comp_get_target = comp.create_component_from_func(
                                            get_target,
                                            base_image="gcr.io/dots-stock/python-img-v5.2"
                                            )

comp_get_tech_indi = comp.create_component_from_func(
                                            get_tech_indi,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            packages_to_install=["stockstats", "scikit-learn"]
                                            )

comp_get_full_tech_indi = comp.create_component_from_func(
                                            get_full_tech_indi,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            )

comp_get_ml_dataset = comp.create_component_from_func(
                                            get_ml_dataset,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            )                                            

comp_get_prediction_04 = comp.create_component_from_func(
                                            get_prediction_04,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            packages_to_install=['catboost', 'scikit-learn', 'ipywidgets']
                                            )    

        
comp_get_update_price_daily = comp.create_component_from_func(
                                            update_price_daily,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            )       

#########################################
# create pipeline #######################
#########################################
job_file_name='gb-pipeline-prediction-daily.json'
@dsl.pipeline(
  name=job_file_name.split('.json')[0],
  pipeline_root=PIPELINE_ROOT
)    
def create_awesome_pipeline():

        op_set_default = comp_set_default('pred')

        with dsl.Condition(op_set_default.outputs['isBusinessDay'] == 'yes'):

                op_get_market_info = comp_get_market_info(
                        date_ref=op_set_default.outputs['date_ref'],
                        n_days=op_set_default.outputs['n_days'])

                op_get_bros = comp_get_bros(
                        date_ref=op_set_default.outputs['date_ref'],
                        n_days=op_set_default.outputs['n_days'])

                op_get_features = comp_get_features(
                        market_info_dataset = op_get_market_info.outputs['market_info_dataset'],
                        bros_univ_dataset = op_get_bros.outputs['bros_univ_dataset']
                )

                op_get_adj_price_01 = comp_get_adj_price(
                        start_index=0,
                        end_index=600,
                        market_info_dataset = op_get_market_info.outputs['market_info_dataset'])

                op_get_adj_price_02 = comp_get_adj_price(
                        start_index=600,
                        end_index=1200,
                        market_info_dataset = op_get_market_info.outputs['market_info_dataset'])
                
                op_get_adj_price_03 = comp_get_adj_price(
                        start_index=1200,
                        end_index=1800,
                        market_info_dataset = op_get_market_info.outputs['market_info_dataset'])
                
                op_get_adj_price_04 = comp_get_adj_price(
                        start_index=1800,
                        end_index=2400,
                        market_info_dataset = op_get_market_info.outputs['market_info_dataset'])
                
                op_get_adj_price_05 = comp_get_adj_price(
                        start_index=2400,
                        end_index=4000,
                        market_info_dataset = op_get_market_info.outputs['market_info_dataset'])
                
                op_get_full_adj_price = comp_get_full_adj_price(
                        adj_price_dataset01 = op_get_adj_price_01.outputs['adj_price_dataset'],
                        adj_price_dataset02 = op_get_adj_price_02.outputs['adj_price_dataset'],
                        adj_price_dataset03 = op_get_adj_price_03.outputs['adj_price_dataset'],
                        adj_price_dataset04 = op_get_adj_price_04.outputs['adj_price_dataset'],
                        adj_price_dataset05 = op_get_adj_price_05.outputs['adj_price_dataset'])

                op_get_target = comp_get_target(
                        full_adj_prices_dataset = op_get_full_adj_price.outputs['full_adj_prices_dataset'])

                op_get_tech_indi_01 = comp_get_tech_indi(
                        adj_price_dataset = op_get_adj_price_01.outputs['adj_price_dataset'])
                op_get_tech_indi_02 = comp_get_tech_indi(
                        adj_price_dataset = op_get_adj_price_02.outputs['adj_price_dataset'])
                op_get_tech_indi_03 = comp_get_tech_indi(
                        adj_price_dataset = op_get_adj_price_03.outputs['adj_price_dataset'])
                op_get_tech_indi_04 = comp_get_tech_indi(
                        adj_price_dataset = op_get_adj_price_04.outputs['adj_price_dataset'])
                op_get_tech_indi_05 = comp_get_tech_indi(
                        adj_price_dataset = op_get_adj_price_05.outputs['adj_price_dataset'])

                op_get_full_tech_indi = comp_get_full_tech_indi(
                        tech_indi_dataset01 = op_get_tech_indi_01.outputs['techini_dataset'],
                        tech_indi_dataset02 = op_get_tech_indi_02.outputs['techini_dataset'],
                        tech_indi_dataset03 = op_get_tech_indi_03.outputs['techini_dataset'],
                        tech_indi_dataset04 = op_get_tech_indi_04.outputs['techini_dataset'],
                        tech_indi_dataset05 = op_get_tech_indi_05.outputs['techini_dataset'])

                op_get_ml_dataset = comp_get_ml_dataset(
                        features_dataset = op_get_features.outputs['features_dataset'],
                        target_dataset = op_get_target.outputs['target_dataset'],
                        tech_indi_dataset = op_get_full_tech_indi.outputs['full_tech_indi_dataset'],
                )

                op_get_prediction_04 = comp_get_prediction_04(
                        ml_dataset = op_get_ml_dataset.outputs['ml_dataset'],
                        bros_univ_dataset = op_get_bros.outputs['bros_univ_dataset']
                )

                op_get_update_price = comp_get_update_price_daily(
                        predictions = op_get_prediction_04.outputs['predictions'],
                )


        


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
#   enable_caching= True,
#   pipeline_root=PIPELINE_ROOT
# )

response = api_client.create_schedule_from_job_spec(
    job_spec_path=job_file_name,
    schedule="30 14 * * 1-5",
    time_zone="Asia/Seoul",
    enable_caching = False,
)