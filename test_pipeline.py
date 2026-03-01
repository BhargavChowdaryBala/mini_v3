"""
Quick test script to verify the enhanced ANPR pipeline.
Tests the /api/process_uploaded_video endpoint and polls for results.
"""
import requests
import time
import json

BASE_URL = "http://127.0.0.1:5000"
VIDEO_PATH = "bus.mp4"

def test_health():
    print("--- [1] Health Check ---")
    r = requests.get(f"{BASE_URL}/health")
    print(f"Status: {r.status_code} | {r.json()}")
    return r.status_code == 200

def test_video_analysis():
    print("\n--- [2] Triggering Video Analysis ---")
    r = requests.post(
        f"{BASE_URL}/api/process_uploaded_video",
        json={"file_path": VIDEO_PATH}
    )
    print(f"Response: {r.status_code}")
    print(f"Raw Body: {r.text}")
    if r.status_code != 200:
        print("ERROR: Could not start video analysis!")
        return False
    
    print("\n--- [3] Polling Upload Status ---")
    for i in range(360):  # Poll for up to 180 seconds
        time.sleep(0.5)
        s = requests.get(f"{BASE_URL}/api/upload_status").json()
        print(f"  [{i+1}] Status: {s.get('status')} | Progress: {s.get('progress')}%")
        if s.get("status") in ("complete", "error"):
            break
    
    return s.get("status") == "complete"

def test_logs():
    print("\n--- [4] Checking Activity Logs ---")
    r = requests.get(f"{BASE_URL}/api/logs")
    if r.status_code == 200:
        logs = r.json()
        print(f"Total log entries: {len(logs)}")
        for entry in logs[:5]:
            print(f"  Plate: {entry.get('registration_number', 'N/A')} | Status: {entry.get('status', 'N/A')} | Time: {entry.get('timestamp', 'N/A')}")
    else:
        print("Could not fetch logs.")

if __name__ == "__main__":
    if test_health():
        ok = test_video_analysis()
        if ok:
            test_logs()
        else:
            print("\nVideo analysis did not complete successfully.")
    else:
        print("Backend is not running. Start it with: python backend/main.py")
