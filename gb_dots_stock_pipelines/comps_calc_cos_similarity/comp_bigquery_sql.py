def create_market_snap_top30_eval(date_ref: str):

  PROJECT_ID = 'dots-stock'
  from google.cloud import bigquery
  client = bigquery.Client(PROJECT_ID)
  sql = f"""
    -- create eval by date 상승가능성 높거나 낮은 것들로만
  CREATE TEMP TABLE tmp_pattern_eval_by_date AS (
  SELECT
    source_code,
    variable AS date_passed,
    ret_p_d0_2_count,
    ret_p_d0_2_mean,
    ud_r_d0_2,
    kernel_size,
    ABS(ud_r_d0_2 - 0.5) AS abs_updn,
    ROW_NUMBER() OVER (PARTITION BY source_code ORDER BY ABS(ud_r_d0_2 - 0.5) DESC,
      variable ASC) AS rank
  FROM
    `dots-stock.red_lion.pattern_eval_{date_ref}`
  WHERE
    variable IN ('d1_2',
      'd3_2',
      'd5_2')
    AND ret_p_d0_2_count > 9
  ORDER BY
    abs_updn DESC,
    source_code );
  -- 임시로 만든 테이블을 이용해 array 만듬
  CREATE TEMP TABLE top2_each_code AS
SELECT
  source_code,
  ARRAY_AGG(ret_p_d0_2_count) AS ret_p_d0_2_count,
  ARRAY_AGG(date_passed) AS date_passed,
  ARRAY_AGG(ud_r_d0_2) AS ud_r_d0_2,
  ARRAY_AGG(rank) AS rank,
  ARRAY_AGG(kernel_size) AS kernel_size
FROM (
  SELECT
    *
  FROM
    tmp_pattern_eval_by_date
  ORDER BY
    date_passed)
GROUP BY
  source_code ;
  -- market snapshot 에 pattern eval 결과를 붙임
CREATE OR REPLACE TABLE
  `red_lion.market_snapshot_top30_eval_{date_ref}` AS (
  SELECT
    *
  FROM
    `dots-stock.red_lion.market_snapshot_top30_{date_ref}` AS lt
  LEFT JOIN
    top2_each_code AS rt
  ON
    lt.Code = rt.source_code )"""
  query_job = client.query(sql)