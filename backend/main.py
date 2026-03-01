import os
# Fix DLL conflicts between Torch and Paddle (WinError 127)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Aggressively disable oneDNN/MKLDNN to prevent "OneDnnContext does not have the input Filter" crash on Windows
# These MUST be set before any other imports (especially paddle and torch)
os.environ['PADDLE_ONEDNN_DISABLE'] = '1'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_enable_mkldnn_bfloat16'] = '0'
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
# os.environ['OMP_NUM_THREADS'] = '1' # Limit threads to reduce contention on some Windows systems - REMOVED

# Import torch/ultralytics FIRST to avoid DLL conflicts
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import torch
from processor import BusProcessor, VideoUploadProcessor, apply_professional_restoration, extract_indian_number_plate
from database import get_recent_logs
import threading
import time
import os
import numpy as np
import base64
import re

app = Flask(__name__)
CORS(app)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()})

# For demo purposes, we use bus.mp4 as the "Live" source. 
# Set to 0 for real webcam.
VIDEO_SOURCE = "bus.mp4" 

# Initialize processors (Isolated)
# Line config for Live/File Pipeline (Adjusted for bus.mp4)
LIVE_LINE_POS = 0.5
LIVE_LINE_DIR = 'vertical'

print("Initializing BusProcessor (File-based Stream)...")
processor = BusProcessor(line_position=LIVE_LINE_POS, line_direction=LIVE_LINE_DIR)

print("Initializing VideoUploadProcessor (Manual Upload)...")
# For manual uploads, a horizontal line at 60% height is often standard
upload_processor = VideoUploadProcessor(line_position=0.6, line_direction='horizontal')
print("Processors Ready.")

# Global state
last_frame = None
lock = threading.Lock()
current_source = VIDEO_SOURCE
# Upload Mode State (Isolated)
upload_status = {"status": "idle", "progress": 0}
last_upload_frame = None
upload_lock = threading.Lock()
live_thread_running = True

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file"}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)
        
        return jsonify({"status": "success", "file_path": file_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/process_uploaded_video', methods=['POST'])
def process_uploaded_video():
    data = request.json or {}
    file_path = data.get('file_path')
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "Invalid file path"}), 400

    def run_analysis():
        global upload_status, last_upload_frame
        with upload_lock:
            if upload_status["status"] == "processing":
                print("Warning: Analysis already in progress. Denying new request.")
                return
            upload_status = {"status": "processing", "progress": 0}
            last_upload_frame = None
        
        def update_progress(p):
            with upload_lock:
                upload_status["progress"] = p

        def update_frame(frame):
            global last_upload_frame
            with upload_lock:
                # Crucial: copy frame to avoid memory corruption during async MJPEG encoding
                last_upload_frame = frame.copy()

        try:
            print(f"[Upload] Starting processing for {file_path}")
            upload_processor.process_video(file_path, progress_callback=update_progress, frame_callback=update_frame)
            with upload_lock:
                upload_status["status"] = "complete"
                upload_status["progress"] = 100
            print("[Upload] Processing finished successfully.")
        except Exception as e:
            print(f"!!! Upload Process Error: {e}")
            with upload_lock:
                upload_status["status"] = "error"

    threading.Thread(target=run_analysis, daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/api/upload_status', methods=['GET'])
def get_upload_status():
    with upload_lock:
        return jsonify(upload_status)

def generate_upload_frames():
    global last_upload_frame
    # Wait up to 5 seconds for processing to actually start
    for i in range(50): 
        if last_upload_frame is not None: break
        time.sleep(0.1)

    while True:
        with upload_lock:
            # If processing is error/idle and no frame yet, stop stream to avoid hanging browser
            if last_upload_frame is None and upload_status["status"] in ["error", "idle"]:
                break

            if last_upload_frame is None:
                # Create a simple "Processing..." placeholder frame in-memory if no frame yet
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "INITIALIZING AI PIPELINE...", (100, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 242, 254), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
            else:
                ret, buffer = cv2.imencode('.jpg', last_upload_frame)
            
            if not ret:
                time.sleep(0.1)
                continue
            frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Limit frame rate for the feed to save bandwidth/CPU
        time.sleep(0.06) # ~16 FPS is enough for visual feedback

@app.route('/upload_video_feed')
def upload_video_feed():
    return Response(generate_upload_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================
# DISABLE LIVE CAMERA FUNCTIONALITY
# ==========================================
# Live webcam and RTSP capture code are disabled per user requirements.
# The system now uses a file-based pipeline for higher reliability and proximity-based OCR.

def background_capture():
    """Background thread to capture from LIVE camera and process frames."""
    global last_frame, live_thread_running
    print(f"[Live] Background capture thread started (Source: {VIDEO_SOURCE})")
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    is_file = isinstance(VIDEO_SOURCE, str)
    
    while live_thread_running:
        ret, frame = cap.read()
        if not ret:
            if is_file:
                # Loop video for continuous live demo
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[Live] Failed to read from camera. Retrying...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(VIDEO_SOURCE)
                continue
            
        # 1. Process Frame using core intelligence
        # This handles detection, tracking, crossing, and OCR triggering
        annotated_frame = processor.process_frame(frame)
        
        # 2. Update global last_frame for MJPEG streaming
        with lock:
            last_frame = annotated_frame.copy()
            
        # Limit CPU usage (Live processing at ~15-20 FPS is sufficient)
        time.sleep(0.01)

    cap.release()
    print("[Live] Background capture thread stopped.")

# [DISABLED] Start the background capture thread
# threading.Thread(target=background_capture, daemon=True).start()

def generate_frames():
    global last_frame
    while True:
        with lock:
            if last_frame is None:
                time.sleep(0.1)
                continue
            
            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', last_frame)
            if not ret:
                time.sleep(0.1)
                continue
            frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Limit frame rate for the feed
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/list_cameras')
def list_cameras():
    """Mock camera list for UI compatibility."""
    return jsonify([{"id": 0, "name": "Primary Camera (CCTV)"}])

@app.route('/api/reset_camera', methods=['POST'])
def reset_camera():
    """Reset camera state (Mock)."""
    return jsonify({"status": "success"})

@app.route('/api/logs')
def logs():
    try:
        source = request.args.get('source')
        data = get_recent_logs(source=source)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({"status": processor.get_status()})

@app.route('/api/check_image', methods=['POST'])
def check_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files['image']
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Resize large images to prevent "OneDnnContext" errors and speed up processing
        h, w = img.shape[:2]
        max_dim = 1600
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Start Metrics
        with processor.lock: # Ensure thread-safe model access
            start_yolo = time.time()
            results = processor.plate_model.predict(source=img, conf=0.4, verbose=False)
            yolo_time = (time.time() - start_yolo) * 1000 # ms
            
            plate_img_b64 = None
            plate_text = "NOT DETECTED"
            confidence = 0
            raw_texts = []
            
            # Extract plate crop
            if len(results[0].boxes) > 0:
                box = results[0].boxes.xyxy[0].cpu().numpy()
                confidence = float(results[0].boxes.conf[0].cpu().item())
                x1, y1, x2, y2 = map(int, box)
                plate_crop = img[y1:y2, x1:x2]
                
                # Check for zero-size crops
                if plate_crop.size == 0:
                    return jsonify({"plate_text": "NOT DETECTED", "confidence": 0, "plate_image": None, "metrics": {"yolo": round(yolo_time, 2), "ocr": 0}})

                # Apply Professional ISP Preprocessing (Returns [pass_a, pass_b, pass_c])
                plate_versions = apply_professional_restoration(plate_crop)
                # Use Pass A (Grayscale) for initial display and OCR
                plate_crop_enhanced = plate_versions[0]
                
                _, buffer = cv2.imencode('.jpg', plate_crop_enhanced)
                plate_img_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Start OCR Metrics
                start_ocr = time.time()
                ocr_time = 0 
                
                try:
                    # Directly pass numpy array to PaddleOCR
                    result = processor.ocr.ocr(plate_crop_enhanced, cls=True)
                    ocr_time = (time.time() - start_ocr) * 1000 # ms
                    
                    if result and result[0]:
                        raw_texts = []
                        for idx in range(len(result)):
                            for line in result[idx]:
                                raw_texts.append(line[1][0])
                        
                        plate_text = extract_indian_number_plate(raw_texts)
                except Exception as e:
                    print("!!! OCR CRITICAL ERROR !!!")
                    import traceback
                    traceback.print_exc()
                    print(f"Exception details: {str(e)}")
                    ocr_time = (time.time() - start_ocr) * 1000
                    plate_text = "ERR_OCR"
            else:
                ocr_time = 0

        return jsonify({
            "plate_text": plate_text,
            "confidence": round(confidence * 100, 1),
            "plate_image": plate_img_b64,
            "raw_texts": raw_texts,
            "metrics": {
                "yolo": round(yolo_time, 2),
                "ocr": round(ocr_time, 2)
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Disable reloader to prevent multiple thread initializations
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)
