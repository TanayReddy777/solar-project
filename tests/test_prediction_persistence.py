import unittest
from unittest.mock import Mock, patch

from bson import ObjectId

import app


class PredictPersistenceTests(unittest.TestCase):
    def test_predict_saves_inputs_and_prediction_to_forecasts(self):
        record_id = ObjectId("507f1f77bcf86cd799439011")
        collection = Mock()
        collection.insert_one.return_value.inserted_id = record_id
        request = app.Reading(
            plant_id=4136001,
            date_time="13-06-2020 13:30",
            ambient_temperature=31.9102,
            module_temperature=47.1552,
            irradiation=0.7514,
        )

        with patch.object(app, "predict_kw", return_value=123.4), patch.object(
            app.db, "forecasts", collection
        ):
            response = app.predict(request)

        saved = collection.insert_one.call_args.args[0]
        self.assertEqual(saved["plant_id"], 4136001)
        self.assertEqual(saved["date_time"], "13-06-2020 13:30")
        self.assertEqual(saved["irradiation"], 0.7514)
        self.assertEqual(saved["predicted_ac_power_kw"], 123.4)
        self.assertNotIn("ac_power", saved)
        self.assertIn("created_at", saved)
        self.assertIn("updated_at", saved)
        self.assertEqual(response["id"], str(record_id))
        self.assertEqual(response["predicted_ac_power_kw"], 123.4)


if __name__ == "__main__":
    unittest.main()

