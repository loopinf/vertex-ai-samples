# Preference
run_no = '05'

start_date = '20201102'
ml_dataset_name = 'ml_dataset_20210914_260.pkl'
bros_dataset_name = 'bros_dataset_20210914_260'

# Preset for GCP
PROJECT_ID = "dots-stock"  # @param {type:"string"}
REGION = "us-central1"  # @param {type:"string"}
USER = "shkim01"  # <---CHANGE THIS
BUCKET_NAME = "gs://pipeline-dots-stock"  # @param {type:"string"}
PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root/{USER}"


from kfp import dsl
from kfp.dsl.io_types import Output
from kfp.v2 import compiler
import kfp.components as comp
from kfp.v2.google.client import AIPlatformClient

from comps_model_backtesting.comp_load_dataset import get_dataset
from comps_model_backtesting.comp_conditioning_dataset import conditioning_dataset

from comps_model_backtesting.comp_ml_m19_8 import get_ml_op
tested_model = 'm19-8' # Should Match Left and Above !!!!!

from comps_model_backtesting.comp_add_prices_n_returns import add_prices_n_returns
from comps_model_backtesting.comp_calc_final_return import calc_returns


comp_data_load = comp.create_component_from_func_v2(
                                            get_dataset,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            )

comp_conditioning_dataset = comp.create_component_from_func_v2(
                                            conditioning_dataset,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            )

comp_ml_op = comp.create_component_from_func_v2(
                                            get_ml_op,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            packages_to_install=['catboost', 'scikit-learn', 'ipywidgets']
                                            )

comp_add_price_n_return = comp.create_component_from_func_v2(
                                            add_prices_n_returns,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            )

comp_calc_returns = comp.create_component_from_func_v2(
                                            calc_returns,
                                            base_image="gcr.io/dots-stock/python-img-v5.2",
                                            )

# create pipeline 
job_file_name = f'model-backtesting-{tested_model}-{run_no}.json'
@dsl.pipeline(
  name=job_file_name.split('.json')[0],
  pipeline_root=PIPELINE_ROOT
)    
def create_awesome_pipeline():
        op_data_load = comp_data_load(
                                run_no = run_no,
                                ml_dataset_name = ml_dataset_name,
                                bros_dataset_name = bros_dataset_name
                                )

        op_conditioning_dataset = comp_conditioning_dataset(
                                ml_dataset = op_data_load.outputs['ml_dataset']
                                )

        op_comp_ml_op = comp_ml_op(
                                start_date = start_date,
                                pre_processed_dataset = op_conditioning_dataset.outputs['pre_processed_dataset'],
                                bros_dataset = op_data_load.outputs['bros_dataset'],
                                )

        op_add_price_n_return = comp_add_price_n_return(
                                prediction_result = op_comp_ml_op.outputs['prediction_result_dataset']
                                )

        op_calc_returns = comp_calc_returns(
                                price_n_return_updated_dataset = op_add_price_n_return.outputs['price_n_return_updated_dataset']
                                )

    


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