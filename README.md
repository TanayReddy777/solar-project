# Solar power prediction API

Predicts plant AC power from weather and time, and flags unusual readings.
Model: SVR (scikit-learn 1.8.0). Hosted on Render.

Live URL: https://solar-api-876m.onrender.com
Interactive docs: https://solar-api-876m.onrender.com/docs

## Endpoints
- GET /           health check
- POST /predict   predicted AC power (kW)
- POST /deviation compares real and predicted power, flags unusual readings

## Example request (/predict)
{
  "plant_id": 4136001,
  "date_time": "13-06-2020 13:30",
  "ambient_temperature": 31.9102,
  "module_temperature": 47.1552,
  "irradiation": 0.7514
}

Add "ac_power" to the same body for /deviation.
Valid plant_id values: 4135001, 4136001.
