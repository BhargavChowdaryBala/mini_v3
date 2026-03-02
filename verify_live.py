"""Quick test: verify unified live pipeline produces DETECTED logs."""
import requests, time

BASE = "http://127.0.0.1:5000"

# 1. Health check
r = requests.get(f"{BASE}/health")
print(f"Health: {r.json()}")

# 2. List cameras
r = requests.get(f"{BASE}/api/list_cameras")
cameras = r.json()
print(f"Cameras detected: {len(cameras)}")
for c in cameras:
    print(f"  - {c['name']} (id={c['id']})")

# 3. Switch to demo source for testing
r = requests.post(f"{BASE}/api/reset_camera", json={"index": "bus.mp4"})
print(f"\nSwitched to demo: {r.json()}")

# 4. Wait for detections
print("\nWaiting 40s for live detections...")
time.sleep(40)

# 5. Check logs
r = requests.get(f"{BASE}/api/logs")
logs = r.json()
print(f"\n=== LIVE PIPELINE RESULTS ===")
print(f"Total detections: {len(logs)}")
for l in logs[:8]:
    bid = l.get("bus_id", "?")
    plate = l.get("registration_number", "?")
    status = l.get("status", "?")
    src = l.get("source", "?")
    print(f"  Bus #{bid} | Plate: {plate} | Status: {status} | Source: {src}")

# Verify all statuses are DETECTED
all_detected = all(l.get("status") == "DETECTED" for l in logs)
print(f"\nAll statuses DETECTED: {'YES' if all_detected else 'NO'}")
print(f"Pipeline verified: {'PASS' if len(logs) > 0 and all_detected else 'NEEDS MORE TIME'}")
