from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
from processor import BusProcessor
from database import get_recent_logs
import threading
import time
import os
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()})

# Initialize processor
# Use 0 for webcam or a path to a video file / RTSP stream
VIDEO_SOURCE = 0 
print("Initializing BusProcessor...")
processor = BusProcessor()
print("BusProcessor Ready.")

# Global state
last_frame = None
lock = threading.Lock()

class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

    def run(self):
        global last_frame
        print(f"Starting Video Capture from source: {VIDEO_SOURCE}")
        
    def run(self):
        global last_frame
        print(f"Starting Video Capture from source: {VIDEO_SOURCE}")
        
        # Strategy: Try DSHOW first (fixes black screen on some), 
        # but if it produces static or fails, the user can restart or we can try default.
        # For now, let's try the default backend first, and DSHOW only if default fails to open.
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            print("Default capture failed. Trying DSHOW...")
            cap = cv2.VideoCapture(VIDEO_SOURCE + cv2.CAP_DSHOW)
            
        camera_ok = cap.isOpened()
        if not camera_ok:
            print("Error: Could not open video source. Using dummy frames.")
        
        fail_count = 0
        while self.running:
            success = False
            frame = None
            
            if camera_ok:
                success, frame = cap.read()
                if not success:
                    fail_count += 1
                    if fail_count > 5:
                        print("Continuous frame read failure. Switching to dummy fallback.")
                        camera_ok = False
                    time.sleep(0.1)
                    continue
                fail_count = 0
            
            # If we have a frame, verify it's not pure noise (optional heuristic)
            # if success and frame is not None:
            #     if np.std(frame) < 1: # Very flat or static-like
            #         pass 

            if not success or frame is None:
                # Create a dummy frame with troubleshooting info
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                msg = "CAMERA OFFLINE" if not success else "INITIALIZING..."
                cv2.putText(frame, msg, (450, 300), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(frame, f"System Time: {time.strftime('%H:%M:%S')}", (480, 400), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Check if shutter is closed or used by another app", (300, 500), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                time.sleep(0.03)
            
            # Process frame
            processed_frame = processor.process_frame(frame)
            
            # Update global frame
            with lock:
                last_frame = processed_frame
            
            # Small delay to keep CPU sane
            time.sleep(0.01)

# Start camera thread
cam_thread = CameraThread()
cam_thread.start()

def generate_frames():
    global last_frame
    while True:
        with lock:
            if last_frame is None:
                continue
            
            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', last_frame)
            frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Limit frame rate for the feed
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logs')
def logs():
    try:
        data = get_recent_logs()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/process_frame', methods=['POST'])
def process_upload():
    try:
        from flask import request
        import base64
        import numpy as np
        
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "No image data"}), 400
            
        # Decode base64
        format, imgstr = image_data.split(';base64,')
        img_bytes = base64.b64decode(imgstr)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process
        processed = processor.process_frame(frame)
        
        # Encode back to base64
        _, buffer = cv2.imencode('.jpg', processed)
        encoded = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "image": f"data:image/jpeg;base64,{encoded}",
            "status": processor.get_status()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status')
def status():
    return jsonify({"status": processor.get_status()})

import os

if __name__ == '__main__':
    # Disable reloader to prevent multiple thread initializations
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)
