"""Regression tests for zero-irradiance prediction and residual API logic.

Run from the repository root with:
    python -m unittest discover -s tests -v

These tests stub the estimator; they verify API/business logic, not real-model accuracy.
"""
import unittest
from unittest.mock import patch

from pydantic import ValidationError

import app


class StubModel:
    def __init__(self, prediction):
        self.prediction = prediction
        self.called = False
        self.last_frame = None

    def predict(self, frame):
        self.called = True
        self.last_frame = frame.copy()
        return [self.prediction]


def make_conditions(**overrides):
    values = {
        "plant_id": 4136001,
        "date_time": "13-06-2020 13:30",
        "ambient_temperature": 31.0,
        "module_temperature": 45.0,
        "irradiation": 0.7,
    }
    values.update(overrides)
    return app.Conditions(**values)


class PredictionTests(unittest.TestCase):
    def test_zero_irradiation_returns_zero_without_calling_model_for_both_plants(self):
        for plant_id in (4135001, 4136001):
            with self.subTest(plant_id=plant_id):
                estimator = StubModel(9876.0)
                with patch.object(app, "model", estimator):
                    result = app.predict_kw(
                        make_conditions(plant_id=plant_id, irradiation=0.0)
                    )
                self.assertEqual(result, 0.0)
                self.assertFalse(estimator.called)

    def test_zero_ambient_temperature_is_passed_through_to_model(self):
        estimator = StubModel(321.5)
        with patch.object(app, "model", estimator):
            result = app.predict_kw(make_conditions(ambient_temperature=0.0))
        self.assertEqual(result, 321.5)
        self.assertEqual(float(estimator.last_frame.iloc[0]["AMBIENT_TEMPERATURE"]), 0.0)

    def test_zero_module_temperature_is_not_silently_converted_to_zero_power(self):
        estimator = StubModel(321.5)
        with patch.object(app, "model", estimator):
            result = app.predict_kw(make_conditions(module_temperature=0.0))
        self.assertEqual(result, 321.5)
        self.assertEqual(float(estimator.last_frame.iloc[0]["MODULE_TEMPERATURE"]), 0.0)

    def test_negative_irradiation_is_rejected(self):
        with self.assertRaises(ValidationError):
            make_conditions(irradiation=-0.01)

    def test_non_finite_weather_value_is_rejected(self):
        with self.assertRaises(ValidationError):
            make_conditions(ambient_temperature=float("nan"))

    def test_non_finite_model_prediction_is_rejected(self):
        estimator = StubModel(float("nan"))
        with patch.object(app, "model", estimator):
            with self.assertRaises(app.HTTPException) as caught:
                app.predict_kw(make_conditions())
        self.assertEqual(caught.exception.status_code, 500)


class ResidualTests(unittest.TestCase):
    def _assess(self, actual, evidence=None):
        estimator = StubModel(1000.0)
        reading = app.Reading(
            **make_conditions().model_dump(),
            ac_power=actual,
            **(evidence or {}),
        )
        with (
            patch.object(app, "model", estimator),
            patch.object(app, "CENTERS", {}),
            patch.object(app, "REL", None),
        ):
            return app.assess(reading, actual, app.evidence_of(reading))

    def test_actual_equal_to_prediction_has_zero_residual(self):
        result = self._assess(1000.0)
        self.assertEqual(result["residual_kw"], 0.0)

    def test_actual_above_prediction_has_positive_residual(self):
        result = self._assess(1250.0)
        self.assertEqual(result["residual_kw"], 250.0)

    def test_actual_below_prediction_has_negative_residual(self):
        result = self._assess(700.0)
        self.assertEqual(result["residual_kw"], -300.0)

    def test_missing_calibration_is_reported_as_undetermined_not_normal(self):
        result = self._assess(700.0)
        self.assertIsNone(result["deviation"])
        self.assertEqual(result["calibration_status"], "incomplete")
        self.assertIn("undetermined", result["reason"].lower())

    def test_deviation_endpoint_returns_assessment_for_valid_request(self):
        estimator = StubModel(1000.0)
        reading = app.Reading(**make_conditions().model_dump(), ac_power=800.0)
        with (
            patch.object(app, "model", estimator),
            patch.object(app, "CENTERS", {}),
            patch.object(app, "REL", None),
        ):
            result = app.deviation(reading)
        self.assertIsInstance(result, dict)
        self.assertIn("residual_kw", result)
        self.assertIn("calculation", result)
        self.assertEqual(result["residual_kw"], -200.0)

    def _diagnose_with_evidence(self, evidence):
        estimator = StubModel(1000.0)
        reading = app.Reading(
            **make_conditions().model_dump(),
            ac_power=100.0,
            **evidence,
        )
        with (
            patch.object(app, "model", estimator),
            patch.object(app, "THRESHOLDS", {"4136001": 100.0}),
            patch.object(app, "CENTERS", {"4136001": 0.0}),
            patch.object(app, "REL", None),
        ):
            return app.assess(reading, 100.0, app.evidence_of(reading))

    def test_full_zero_output_inverter_evidence(self):
        result = self._diagnose_with_evidence(
            {"inverters_reporting": 22, "inverters_zero_ac": 22}
        )
        self.assertIn("Possible full plant outage", result["likely_causes"])
        self.assertIsNone(result["estimated_inverters_lost"])

    def test_partial_zero_output_inverter_evidence(self):
        result = self._diagnose_with_evidence(
            {"inverters_reporting": 22, "inverters_zero_ac": 2}
        )
        self.assertIn("Possible partial inverter outage", result["likely_causes"])
        self.assertIsNone(result["estimated_inverters_lost"])

    def test_missing_telemetry_is_not_called_confirmed_outage(self):
        result = self._diagnose_with_evidence({"inverters_reporting": 20})
        self.assertIn("Missing inverter telemetry", result["likely_causes"])
        self.assertNotIn("Possible full plant outage", result["likely_causes"])

    def test_zero_irradiance_is_not_classified_as_daytime_outage(self):
        estimator = StubModel(9999.0)
        reading = app.Reading(
            **make_conditions(irradiation=0.0).model_dump(),
            ac_power=0.0,
            inverters_reporting=22,
            inverters_zero_ac=22,
        )
        with (
            patch.object(app, "model", estimator),
            patch.object(app, "CENTERS", {}),
            patch.object(app, "REL", None),
        ):
            result = app.assess(reading, 0.0, app.evidence_of(reading))
        self.assertFalse(result["daytime"])
        self.assertNotIn("Possible full plant outage", result["likely_causes"])


if __name__ == "__main__":
    unittest.main()
