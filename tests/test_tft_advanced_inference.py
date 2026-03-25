from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

from src.service.tft_advanced_service import (
    DEFAULT_BUSINESS_WEEKMASK,
    AdvancedTFTInferenceService,
    DataIntegrityError,
    InferenceError,
)


def build_mock_rows(
    num_records: int = 45,
    base_close: float = 500.0,
    descending: bool = True,
) -> list[dict[str, float | str]]:
    offset = CustomBusinessDay(weekmask=DEFAULT_BUSINESS_WEEKMASK)
    dates = pd.date_range(start="2026-01-01", periods=num_records, freq=offset)

    rows: list[dict[str, float | str]] = []
    for idx, dt in enumerate(dates):
        close_price = float(base_close + idx)
        rows.append(
            {
                "businessDate": dt.date().isoformat(),
                "openPrice": float(close_price - 1.0),
                "highPrice": float(close_price + 2.0),
                "lowPrice": float(close_price - 2.0),
                "closePrice": close_price,
                "totalTradedQuantity": float(1000 + (idx * 10)),
            }
        )

    if descending:
        rows.reverse()
    return rows


class TestAdvancedTFTFeaturePipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AdvancedTFTInferenceService(min_records=45)

    def test_momentum_20d_matches_manual_calculation(self) -> None:
        rows = build_mock_rows(num_records=45, base_close=100.0, descending=True)
        frame = self.service.transform_rows_to_features(symbol="NABIL", raw_rows=rows)

        manual_momentum = (frame.loc[20, "close"] / frame.loc[0, "close"]) - 1.0
        self.assertAlmostEqual(frame.loc[20, "momentum_20d"], manual_momentum, places=10)

    def test_zero_volume_raises_integrity_error(self) -> None:
        rows = build_mock_rows(num_records=45, descending=True)
        rows[10]["totalTradedQuantity"] = 0.0

        with self.assertRaises(DataIntegrityError):
            self.service.transform_rows_to_features(symbol="NABIL", raw_rows=rows)

    def test_large_business_day_gap_raises_integrity_error(self) -> None:
        rows = build_mock_rows(num_records=50, descending=False)
        del rows[20:24]
        rows.reverse()

        with self.assertRaises(DataIntegrityError):
            self.service.transform_rows_to_features(symbol="NABIL", raw_rows=rows)


class TestAdvancedTFTHorizonResponse(unittest.TestCase):
    def test_service_uses_day5_forecast_from_5_step_output(self) -> None:
        rows = build_mock_rows(num_records=45, base_close=500.0, descending=True)
        point_forecast = np.array([[550.0, 551.0, 552.0, 553.0, 554.0]], dtype=float)
        quantile_forecast = np.array(
            [
                [
                    [540.0, 550.0, 560.0],
                    [541.0, 551.0, 561.0],
                    [542.0, 552.0, 562.0],
                    [543.0, 553.0, 563.0],
                    [548.0, 554.0, 560.0],
                ]
            ],
            dtype=float,
        )

        service = AdvancedTFTInferenceService(min_records=45)
        try:
            service._supported_symbols = {"NABIL"}
            service.max_encoder_length = 30
            service.max_prediction_length = 5

            with (
                mock.patch.object(service, "_ensure_model_loaded", return_value=None),
                mock.patch.object(service, "_fetch_symbol_rows", return_value=rows),
                mock.patch.object(
                    service,
                    "_predict_arrays",
                    return_value=(point_forecast, quantile_forecast, [0.1, 0.5, 0.9]),
                ),
            ):
                result = service.predict("NABIL")

            self.assertEqual(result.symbol, "NABIL")
            self.assertEqual(len(result.history.last_7_days), 7)
            self.assertEqual(result.forecast.predicted_magnitude, 554.0)
            self.assertEqual(result.forecast.confidence_interval.low, 548.0)
            self.assertEqual(result.forecast.confidence_interval.high, 560.0)
            self.assertEqual(result.forecast.expected_direction, "UP")

            expected_change = ((554.0 - 544.0) / 544.0) * 100.0
            self.assertAlmostEqual(result.forecast.expected_change_pct, expected_change, places=8)

            as_of_date = pd.Timestamp(result.as_of_date)
            target_date = pd.Timestamp(result.forecast.target_date)
            offset = CustomBusinessDay(weekmask=DEFAULT_BUSINESS_WEEKMASK)
            expected_target = as_of_date + (offset * 5)
            self.assertEqual(target_date.date(), expected_target.date())
        finally:
            service._supported_symbols = set()

    def test_service_rejects_horizon_shorter_than_5(self) -> None:
        rows = build_mock_rows(num_records=45, base_close=500.0, descending=True)
        point_forecast = np.array([[550.0, 551.0, 552.0, 553.0]], dtype=float)
        quantile_forecast = np.array(
            [[[540.0, 550.0, 560.0], [541.0, 551.0, 561.0], [542.0, 552.0, 562.0], [543.0, 553.0, 563.0]]],
            dtype=float,
        )

        service = AdvancedTFTInferenceService(min_records=45)
        service._supported_symbols = {"NABIL"}
        service.max_encoder_length = 30
        service.max_prediction_length = 5

        with (
            mock.patch.object(service, "_ensure_model_loaded", return_value=None),
            mock.patch.object(service, "_fetch_symbol_rows", return_value=rows),
            mock.patch.object(
                service,
                "_predict_arrays",
                return_value=(point_forecast, quantile_forecast, [0.1, 0.5, 0.9]),
            ),
        ):
            with self.assertRaises(InferenceError):
                service.predict("NABIL")


if __name__ == "__main__":
    unittest.main()
