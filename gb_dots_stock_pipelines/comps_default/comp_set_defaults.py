from typing import NamedTuple

def set_defaults()-> NamedTuple(
  'Outputs',
  [
    ('date_ref',str),
    ('n_days', int),
    ('period_extra', int)
  ]):

  import pandas as pd
  from trading_calendars import get_calendar

  today = pd.Timestamp.now('Asia/Seoul').strftime('%Y%m%d')
  today = '20210907'
  period_to_train = 20
  period_extra = 100
  n_days = period_to_train + period_extra

  cal_KRX = get_calendar('XKRX')

  def get_krx_on_dates_start_end(start, end):

      return [date.strftime('%Y%m%d')
              for date in pd.bdate_range(start=start, 
          end=end, freq='C', 
          holidays=cal_KRX.precomputed_holidays)
      ]

  print(f'today : {today}')
  dates_krx_on = get_krx_on_dates_start_end('20210104', today)

  if today in dates_krx_on :
    date_ref = today
  else :
    date_ref = dates_krx_on[-1]
  return (date_ref, n_days, period_extra)