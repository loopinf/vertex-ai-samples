from typing import NamedTuple

def set_defaults()-> NamedTuple(
  'Outputs',
  [
    ('date_ref',str),
    ('n_days', int),
    ('period_extra', int)
  ]):

  date_ref = '20211015'
  period_to_train = 20
  period_extra = 500
  n_days = period_to_train + period_extra

  return (date_ref, n_days, period_extra)