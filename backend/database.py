from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "bus_monitoring"
COLLECTION_NAME = "logs"

# Connect with a timeout to avoid hangs
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client[DB_NAME]
logs_collection = db[COLLECTION_NAME]

def check_db():
    try:
        client.admin.command('ping')
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

def log_event(registration_number, status):
    """
    Logs an entry/exit event to the database.
    """
    log_entry = {
        "registration_number": registration_number,
        "status": status,
        "timestamp": datetime.now()
    }
    try:
        result = logs_collection.insert_one(log_entry)
        print(f"Logged event: {registration_number} - {status} (ID: {result.inserted_id})")
        return True
    except Exception as e:
        print(f"Error logging to database: {e}")
        return False

def get_recent_logs(limit=20):
    """
    Retrieves recent logs for the dashboard.
    """
    logs = list(logs_collection.find().sort("timestamp", -1).limit(limit))
    for log in logs:
        log["_id"] = str(log["_id"])
        log["timestamp"] = log["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    return logs
