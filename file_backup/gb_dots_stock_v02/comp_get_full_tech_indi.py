from kfp.components import InputPath, OutputPath

def get_full_tech_indi(
  tech_indi_dataset01_path: InputPath('DataFrame'),
  tech_indi_dataset02_path: InputPath('DataFrame'),
  tech_indi_dataset03_path: InputPath('DataFrame'),
  tech_indi_dataset04_path: InputPath('DataFrame'),
  tech_indi_dataset05_path: InputPath('DataFrame'),
  full_tech_indi_dataset_path: OutputPath('DataFrame')
):

  import pandas as pd  

  df_01 = pd.read_pickle(tech_indi_dataset01_path)
  df_02 = pd.read_pickle(tech_indi_dataset02_path)
  df_03 = pd.read_pickle(tech_indi_dataset03_path)
  df_04 = pd.read_pickle(tech_indi_dataset04_path)
  df_05 = pd.read_pickle(tech_indi_dataset05_path)
  
  df_full = pd.concat([df_01, df_02, df_03,df_04, df_05])

  df_full.to_pickle(full_tech_indi_dataset_path)