import os
from pymongo import MongoClient

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    if os.getenv("RENDER"):   # Render sets this automatically
        raise RuntimeError("MONGODB_URI is not set in Render's Environment tab")
    MONGODB_URI = "mongodb://localhost:27017"   # local development only

DB_NAME = os.getenv("MONGODB_DB", "solar_db")

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
database = client[DB_NAME]

readings = database["readings"]
forecasts = database["forecasts"]
