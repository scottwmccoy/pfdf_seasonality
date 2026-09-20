"""Guards on the shared definitions consolidated during the 2026-09-20 move.

These exist because consolidating copy-pasted helpers is exactly where a
refactor silently changes a published number. One such change was caught
during the migration and is pinned below.
"""
import numpy as np
import pytest

from pfdf_seasonality.seasons import (
    SEASONS, SEASON_NAMES, MONTH_TO_SEASON, EPS, LAT0,
    normalize, season_of, season_totals, circular, circular_mean_doy,
    circ_diff_days, to_xy, season_columns,
)


def test_seasons_partition_the_year():
    months = sorted(m for ms in SEASONS.values() for m in ms)
    assert months == list(range(1, 13))
    assert list(SEASONS) == ["DJF", "MAM", "JJA", "SON"]
    assert list(SEASON_NAMES) == list(SEASONS)


def test_month_to_season_agrees_with_seasons():
    for season, months in SEASONS.items():
        for m in months:
            assert MONTH_TO_SEASON[m] == season


def test_constants_match_the_originals():
    assert EPS == 1e-4
    assert LAT0 == np.deg2rad(39.0)


def test_season_columns_shapes():
    # 03_build_station_seasonality derives its lowercase names from SEASONS
    assert season_columns()["DJF"] == "djf_frac"
    assert season_columns(suffix="_x", lower=False)["JJA"] == "JJA_x"


def test_normalize_rows_sum_to_one():
    P = np.array([[1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]])
    out = normalize(P)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert (out >= EPS * 0.99).all(), "the floor keeps one odd event from vetoing a model"


def test_normalize_all_zero_row_is_nan():
    assert np.isnan(normalize(np.zeros((1, 12)))).all()


def test_season_of_picks_the_right_season():
    P = np.zeros((1, 12))
    P[0, 6] = 1.0                      # July -> JJA
    name, share = season_of(P)
    assert name[0] == "JJA" and share[0] == pytest.approx(1.0)
    assert season_totals(P).shape == (1, 4)


def test_circular_mean_doy_uses_365_not_365_25():
    """Pinned: the original used 365.0. Changing it shifts every clock number.

    A distribution concentrated in a single month must land on that month's
    mid-point under a 365.0-day year, not a 365.25-day one.
    """
    P = np.zeros((1, 12))
    P[0, 0] = 1.0                      # January, theta = 2*pi*0.5/12
    expected = (0.5 / 12) * 365.0
    assert circular_mean_doy(P)[0] == pytest.approx(expected, abs=1e-9)


def test_circular_reports_concentration_and_p():
    mean_doy, R, p = circular([10, 11, 12, 13, 14])
    assert 0 < mean_doy < 365.25
    assert R > 0.99                    # tightly clustered
    assert 0 <= p <= 1
    _, R_spread, _ = circular(np.linspace(0, 364, 48))
    assert R_spread < 0.05             # uniform around the year


def test_circ_diff_days_wraps_at_new_year():
    assert circ_diff_days(360, 5) == pytest.approx(10.25, abs=0.3)
    assert circ_diff_days(100, 100) == 0


def test_to_xy_is_monotonic_and_metric():
    xy = to_xy([39.0, 40.0], [-120.0, -120.0])
    assert xy.shape == (2, 2)
    # one degree of latitude is about 111 km
    assert abs(xy[1, 1] - xy[0, 1]) == pytest.approx(111.2, abs=1.0)
