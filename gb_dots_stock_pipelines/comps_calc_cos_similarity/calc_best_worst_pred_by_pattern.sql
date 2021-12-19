CREATE TEMP TABLE tmp_pattern_eval_by_date AS ( (
  WITH
    ordered_ud_r_d0_2_per_code AS (
    SELECT
      source_code,
      variable AS date_passed,
      ret_p_d0_2_count,
      ret_p_d0_2_mean,
      ud_r_d0_2,
      kernel_size,
      ROW_NUMBER() OVER (PARTITION BY source_code ORDER BY ud_r_d0_2 DESC, variable ASC) AS rank
    FROM (
      SELECT
        *
      FROM
        `dots-stock.red_lion.pattern_eval_20211217`
      WHERE
        ud_r_d0_2 > 0.65))
  SELECT
    *,
    "up" AS type
  FROM
    ordered_ud_r_d0_2_per_code
  WHERE
    rank IN (1,
      2)
  ORDER BY
    ud_r_d0_2 DESC)
UNION ALL (
    -- 내린쪽만 하기
  WITH
    ordered_ud_r_d0_2_per_code AS (
    SELECT
      source_code,
      variable AS date_passed,
      ret_p_d0_2_count,
      ret_p_d0_2_mean,
      ud_r_d0_2,
      kernel_size,
      ROW_NUMBER() OVER (PARTITION BY source_code ORDER BY ud_r_d0_2 ASC, variable ASC) AS rank
    FROM (
      SELECT
        *
      FROM
        `dots-stock.red_lion.pattern_eval_20211217`
      WHERE
        ud_r_d0_2 < 0.35))
  SELECT
    *,
    "down" AS type
  FROM
    ordered_ud_r_d0_2_per_code
  WHERE
    rank IN (1,
      2)
  ORDER BY
    ud_r_d0_2 DESC)); CREATE temp TABLE top2_each_code AS
SELECT
source_code,
type,
ret_p_d0_2_count,
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
source_code,
type,
ret_p_d0_2_count;
CREATE OR REPLACE TABLE
red_lion.market_snapshot_top30_eval_20211217 AS (
SELECT
  *
FROM
  `dots-stock.red_lion.market_snapshot_top30_20211217` AS lt
LEFT JOIN
  top2_each_code AS rt
ON
  lt.Code = rt.source_code )