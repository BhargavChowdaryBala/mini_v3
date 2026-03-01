from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "bus_monitoring"
COLLECTION_NAME = "logs"

# Connect with a timeout to avoid hangs
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    # Trigger a quick check
    client.admin.command('ping')
    db = client[DB_NAME]
    logs_collection = db[COLLECTION_NAME]
    DB_CONNECTED = True
    print("Successfully connected to MongoDB.")
except Exception as e:
    print(f"Warning: Database connection failed: {e}")
    DB_CONNECTED = False
    # Mock storage for session-only logs if DB fails
    MOCK_LOGS = []

def check_db():
    return DB_CONNECTED

def log_event(registration_number, status, source="live", bus_id=None):
    """
    Logs an entry/exit event to the database. Unique per (bus_id, source).
    """
    now = datetime.now()
    # Cast bus_id to standard python int if it's a numpy type (prevents MongoDB serialization error)
    if bus_id is not None:
        try:
            bus_id = int(bus_id)
        except:
            pass

    log_entry = {
        "registration_number": registration_number,
        "status": status,
        "timestamp": now,
        "source": source,
        "bus_id": bus_id
    }
    
    if DB_CONNECTED:
        try:
            # Use bus_id and source as the unique identifier for a detection event
            # to allow multiple buses with 'No Plate' or the same plate (unlikely but possible)
            filter_query = {"source": source}
            if bus_id is not None:
                filter_query["bus_id"] = bus_id
            else:
                filter_query["registration_number"] = registration_number

            result = logs_collection.update_one(
                filter_query,
                {"$set": log_entry},
                upsert=True
            )
            
            identifier = bus_id if bus_id else registration_number
            if result.upserted_id:
                print(f"Logged NEW {source} event (ID:{identifier}): {registration_number} - {status}")
            else:
                print(f"Updated EXISTING {source} event (ID:{identifier}): {registration_number} - {status}")
            return True
        except Exception as e:
            print(f"Error logging to database: {e}")
            return False
    else:
        # Fallback to in-memory logs with unique check per source
        existing_idx = next((i for i, log in enumerate(MOCK_LOGS) 
                             if log["registration_number"] == registration_number and log["source"] == source), None)
        if existing_idx is not None:
            MOCK_LOGS[existing_idx]["timestamp"] = now
            MOCK_LOGS[existing_idx]["status"] = status
            print(f"Updated EXISTING {source} event (MEMORY): {registration_number} - {status}")
        else:
            log_entry["_id"] = f"mock_{len(MOCK_LOGS)}"
            MOCK_LOGS.insert(0, log_entry)
            print(f"Logged NEW {source} event (MEMORY): {registration_number} - {status}")
        
        if len(MOCK_LOGS) > 200: MOCK_LOGS.pop()
        return True

def get_recent_logs(limit=20, source=None):
    """
    Retrieves recent logs for the dashboard from DB or memory.
    Supports filtering by source prefix (e.g., 'upload' or 'live').
    """
    if DB_CONNECTED:
        try:
            query = {}
            if source:
                # Case-insensitive prefix match for source
                query["source"] = {"$regex": f"^{source}", "$options": "i"}
            
            logs = list(logs_collection.find(query).sort("timestamp", -1).limit(limit))
            for log in logs:
                log["_id"] = str(log["_id"])
                log["timestamp"] = log["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            return logs
        except Exception as e:
            print(f"DB Fetch Error: {e}")
            return []
    else:
        # Return mock logs with optional source filtering
        filtered_logs = MOCK_LOGS
        if source:
            filtered_logs = [log for log in MOCK_LOGS if log.get("source", "").lower().startswith(source.lower())]
            
        # Return mock logs with formatted timestamp
        formatted_logs = []
        for log in filtered_logs[:limit]:
            entry = log.copy()
            if isinstance(entry["timestamp"], datetime):
                entry["timestamp"] = entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            formatted_logs.append(entry)
        return formatted_logs
