"""MongoDB connection. The address is read from the MONGODB_URI environment variable,
so no password is ever written in the code or pushed to GitHub."""
import os

from pymongo import MongoClient

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")   # local MongoDB by default
DB_NAME = os.getenv("MONGODB_DB", "solar_db")

# MongoClient connects lazily, so the API still starts if the database is briefly unreachable
client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
database = client[DB_NAME]

readings = database["readings"]      # Plant Engineer: actual readings + model check + notes
forecasts = database["forecasts"]    # DISCOM: what-if forecasts + planning notes