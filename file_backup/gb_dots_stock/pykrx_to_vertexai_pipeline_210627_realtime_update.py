# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd

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
    base_image="gcr.io/dots-stock/python-img-v4.0",
)
def check()-> str:
    return 'Helllo'


@component(
    base_image="gcr.io/dots-stock/python-img-v4.0",
)
def get_univ(
    s_univ_dataset: Output[Dataset]
) -> str:
    import pickle
    import os
    import pandas as pd
    today = '20210811'
    file_path = "/gcs/pipeline-dots-stock/s_univ_top30_theDay_and_bros"
    file_name = f"s_univ_top30_theDay_and_bros_{today}.pickle"
    
    full_path = os.path.join(file_path, file_name)
    # full_path = 'test.pickle'
    with open(full_path, 'rb') as f:
        dict_s_univ = pickle.load(f)
    #%%

    dict_s_univ
    #%%
    dic2 = { stock: date for date, stocks in dict_s_univ.items() for stock in stocks}
    df = pd.DataFrame.from_dict(dic2, orient='index', columns=['date']).reset_index().rename(columns={'index':'code', '0':''})
    df.to_csv(s_univ_dataset.path)

@component()
def get_feature() -> str:
    pass

@component()
def get_target() -> str:
    pass
# str_now = pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d-%H%M%S')
job_file_name = f'testing-aaa.json' # job_file_name 에 대문자 쓰지마!
@dsl.pipeline(
    # name="dots-stock-update-price-every-5min",
    name=f"testing-aaa",
    # description="",
    pipeline_root=PIPELINE_ROOT,
)
def intro_pipeline():
    result = get_univ()

compiler.Compiler().compile(
    pipeline_func=intro_pipeline, package_path=job_file_name
    )

api_client = AIPlatformClient(
    project_id=PROJECT_ID,
    region=REGION,
)

response = api_client.create_run_from_job_spec(
    job_spec_path=job_file_name,
    enable_caching = False,
    pipeline_root=PIPELINE_ROOT  # this argument is necessary if you did not specify PIPELINE_ROOT as part of the pipeline definition.
)

# """# Add Scheduler"""

# # adjust time zone and cron schedule as necessary
# response = api_client.create_schedule_from_job_spec(
#     job_spec_path=job_file_name,
#     schedule="21-59/10 9 * * 1-5",
#     time_zone="Asia/Seoul",  # change this as necessary
#     enable_caching = False,
#     # parameter_values={"text": "To the GB"},
#     # pipeline_root=PIPELINE_ROOT  # this argument is necessary if you did not specify PIPELINE_ROOT as part of the pipeline definition.
# )

# # adjust time zone and cron schedule as necessary
# response = api_client.create_schedule_from_job_spec(
#     job_spec_path=job_file_name,
#     schedule="00-59/10 10-14 * * 1-5",
#     time_zone="Asia/Seoul",  # change this as necessary
#     enable_caching = False,
#     # parameter_values={"text": "To the GB"},
#     # pipeline_root=PIPELINE_ROOT  # this argument is necessary if you did not specify PIPELINE_ROOT as part of the pipeline definition.
# )

# # adjust time zone and cron schedule as necessary
# response = api_client.create_schedule_from_job_spec(
#     job_spec_path=job_file_name,
#     schedule="00-35/10 15 * * 1-5",
#     time_zone="Asia/Seoul",  # change this as necessary
#     enable_caching = False,
#     # parameter_values={"text": "To the GB"},
#     # pipeline_root=PIPELINE_ROOT  # this argument is necessary if you did not specify PIPELINE_ROOT as part of the pipeline definition.
# )

# # adjust time zone and cron schedule as necessary
# response = api_client.create_schedule_from_job_spec(
#     job_spec_path=job_file_name,
#     schedule="35 15 * * 1-5",
#     time_zone="Asia/Seoul",  # change this as necessary
#     enable_caching = False
#     # parameter_values={"text": "To the GB"},
#     # pipeline_root=PIPELINE_ROOT  # this argument is necessary if you did not specify PIPELINE_ROOT as part of the pipeline definition.
# )

# """## Scheduler Note
# 1. "2-59/5 17-23 * * *" --> corr value 업데이트 1차


# 나중에 해볼꺼
# - https://towardsdatascience.com/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3
# """