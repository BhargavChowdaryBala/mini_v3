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
from processor import BusProcessor, VideoUploadProcessor, apply_professional_restoration, extract_indian_number_plate, deskew_plate, apply_padding
from database import get_recent_logs
import threading
import time
import os
import numpy as np
import base64
import re

# NEW: Capture camera names on Windows
if os.name == 'nt':
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        FilterGraph = None
else:
    FilterGraph = None

app = Flask(__name__)
CORS(app)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()})

# For demo purposes, we use bus.mp4 as the default "Live" source. 
# Set to 0 for real webcam.
def detect_cameras():
    """Detects available hardware cameras with friendly names on Windows."""
    available = []
    
    # 1. Capture names using pygrabber on Windows
    device_names = []
    if os.name == 'nt' and FilterGraph:
        try:
            graph = FilterGraph()
            device_names = graph.get_input_devices()
        except Exception as e:
            print(f"[System] Warning: Could not resolve camera names: {e}")
            device_names = []

    # 2. Scan cameras and match with names
    for i in range(8): # Check up to 8 slots
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(i)
            if cap.isOpened():
                # Force standard resolution and MJPG format to prevent DirectShow decoding glitches
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, _ = cap.read()
                if ret:
                    friendly_name = f"Camera {i}"
                    if i < len(device_names):
                        friendly_name = device_names[i]
                    
                    available.append({
                        "id": i, 
                        "name": f"{friendly_name.upper()} (SENSOR {i})"
                    })
                cap.release()
        except Exception as e:
            print(f"[System] Skipping camera {i}: {e}")
            continue
            
    # Add demo source
    available.append({"id": "bus.mp4", "name": "SIMULATED CCTV (DEMO FILE)"})
    return available

# Auto-detect on startup to prevent DSHOW crashes later
print("[System] Scanning for hardware cameras (One-time startup check)...")
DETECTED_CAMERAS = detect_cameras()

# Default to first hardware camera if it exists, otherwise use demo
VIDEO_SOURCE = DETECTED_CAMERAS[0]["id"] if DETECTED_CAMERAS else "bus.mp4"

# Initialize processors — BOTH use the SAME pipeline configuration
# Horizontal line at 60% height is the standard for bus monitoring
LIVE_LINE_POS = 0.6
LIVE_LINE_DIR = 'horizontal'

print(f"Initializing BusProcessor for Live Stream (Default Source: {VIDEO_SOURCE})...")
processor = BusProcessor(line_position=LIVE_LINE_POS, line_direction=LIVE_LINE_DIR)

print("Initializing VideoUploadProcessor (Manual Upload Filter Only)...")
upload_processor = VideoUploadProcessor(line_position=LIVE_LINE_POS, line_direction=LIVE_LINE_DIR)
print("Processors Ready.")

# Global State for Live Feed
last_frame = None
lock = threading.Lock()
live_thread_token = 0
live_mode_active = True # NEW: Flag to pause live tracking
live_thread_running = True

# Upload Mode State (Isolated) — RESTORED
upload_status = {"status": "idle", "progress": 0}
last_upload_frame = None
upload_lock = threading.Lock()

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
            try:
                global last_upload_frame
                with upload_lock:
                    h, w = frame.shape[:2]
                    preview_h = 480
                    preview_w = int(w * (preview_h / h))
                    preview_w = (preview_w // 2) * 2
                    preview_frame = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_LINEAR)
                    last_upload_frame = np.ascontiguousarray(preview_frame)
            except Exception as e:
                import traceback
                with open("C:/Users/bharg/Desktop/for_n/debug_update_frame_error.txt", "w") as f_err:
                    traceback.print_exc(file=f_err)
                raise

        try:
            # Basic cleanup: Keep only the most recent few uploads to save space
            files = sorted([os.path.join('uploads', f) for f in os.listdir('uploads') if os.path.isfile(os.path.join('uploads', f))], key=os.path.getmtime)
            if len(files) > 15:
                for f in files[:-15]:
                    try: 
                        os.remove(f)
                    except Exception as e: 
                        print(f"Could not remove old file {f}: {e}")

            print(f"[Upload] Starting processing for {file_path}")
            upload_processor.process_video(file_path, progress_callback=update_progress, frame_callback=update_frame)
            with upload_lock:
                upload_status["status"] = "complete"
                upload_status["progress"] = 100
            print("[Upload] Processing finished successfully.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"!!! Upload Process Error: {e}")
            with open("C:/Users/bharg/Desktop/for_n/debug_error.txt", "w") as f_err:
                traceback.print_exc(file=f_err)
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
    for i in range(50): 
        if last_upload_frame is not None: break
        time.sleep(0.1)

    while True:
        frame_to_use = None
        status_info = None
        
        with upload_lock:
            status_info = upload_status["status"]
            if last_upload_frame is None and status_info in ["error", "idle"]:
                break
            
            if last_upload_frame is not None:
                # Safe copy under lock
                frame_to_use = last_upload_frame.copy()
        
        # Determine what to encode
        if frame_to_use is None:
            # Create a simple "Processing..." placeholder frame in-memory if no frame yet
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "INITIALIZING AI PIPELINE...", (100, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 242, 254), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
        else:
            # Encode OUTSIDE the lock
            ret, buffer = cv2.imencode('.jpg', frame_to_use, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        if not ret:
            time.sleep(0.01)
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # High Speed Manual Analysis Feed
        time.sleep(0.01)

@app.route('/upload_video_feed')
def upload_video_feed():
    return Response(generate_upload_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================
# LIVE MONITORING (Elite-v4 Pipeline)
# ==========================================
# Uses the same BusProcessor.process_frame() as the Video Analyzer.
# Supports hardware cameras (USB/CCTV via CAP_DSHOW on Windows) and file-based demo streams.

def background_capture(token):
    """Background thread to capture from LIVE camera and process frames."""
    global last_frame, live_thread_token
    
    is_file = isinstance(VIDEO_SOURCE, str) and not VIDEO_SOURCE.startswith('rtsp')
    print(f"[Live] Thread {token} active for source {VIDEO_SOURCE}")
    
    cap = None
    if isinstance(VIDEO_SOURCE, int) and os.name == 'nt':
        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"[Live] Hardware camera initialized.")
    else:
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if isinstance(VIDEO_SOURCE, int) and cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    is_file = not isinstance(VIDEO_SOURCE, int)
    
    # Set a session name for the live stream
    source_name = f"cam_{VIDEO_SOURCE}" if isinstance(VIDEO_SOURCE, int) else "demo_live"
    processor.set_session(f"live_{source_name}_{int(time.time())}")
    
    while True:
        if token != live_thread_token:
            print(f"[Live] Thread {token} exiting (Stale).")
            break
        
        # [ELITE RESOURCE MANAGEMENT]
        # Physically release the camera if live mode is inactive
        if not live_mode_active:
            if cap is not None and cap.isOpened():
                try:
                    print("[System] Releasing camera hardware for resource optimization.")
                    cap.release()
                    cap = None
                except Exception as e:
                    print(f"[Warning] Camera release error: {e}")
            time.sleep(0.5)
            continue
        
        # Re-open if we were paused and now active
        if cap is None or not cap.isOpened():
            print("[Debug] Entering camera re-init block...")
            time.sleep(0.2) # Give hardware time to settle
            try:
                if isinstance(VIDEO_SOURCE, int) and os.name == 'nt':
                    print(f"[Debug] Attempting cv2.VideoCapture({VIDEO_SOURCE}, DSHOW)...")
                    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
                else:
                    print(f"[Debug] Attempting cv2.VideoCapture({VIDEO_SOURCE})...")
                    cap = cv2.VideoCapture(VIDEO_SOURCE)
                
                # [CRITICAL FIX] Force resolution and MJPG to prevent DSHOW byte-stride / MJPEG decoding corruption
                if isinstance(VIDEO_SOURCE, int):
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                print("[Live] Camera hardware re-initialized.")
            except Exception as e:
                print(f"[Critical] VideoCapture Exception: {e}")
                continue

        ret, frame = cap.read()
        if not ret:
            if is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                processor.reset()
                continue
            else:
                time.sleep(1)
                continue
        
        # [NEW] Ensure contiguous memory layout immediately after capture
        # This prevents stride/mosaic artifacts in many MJPEG implementations
        frame = np.ascontiguousarray(frame)
            
        # Process and Resize for preview
        try:
            # print(f"[Debug] Processing frame (ID: {id(frame)})...")
            annotated_frame = processor.process_frame(frame)
        except Exception as e:
            print(f"[Critical] processor.process_frame crash: {e}")
            import traceback
            traceback.print_exc()
            continue

        if annotated_frame is None:
            continue

        # print("[Debug] Drawing/Resizing for preview...")
        h, w = annotated_frame.shape[:2]
        preview_h = 480
        preview_w = int(w * (preview_h / h))
        preview_w = (preview_w // 2) * 2 
        preview_frame = cv2.resize(annotated_frame, (preview_w, preview_h), interpolation=cv2.INTER_LINEAR)
        
        # Immediate copy for the global state
        fixed_frame = np.ascontiguousarray(preview_frame)

        with lock:
            last_frame = fixed_frame
            
        # Turbo Performance (1ms yield)
        time.sleep(0.001)

    if cap: cap.release()
    print("[Live] Background capture thread stopped.")

# Start the background capture thread
threading.Thread(target=background_capture, args=(0,), daemon=True).start()

def generate_frames():
    global last_frame
    while True:
        frame_to_use = None
        with lock:
            if last_frame is not None:
                # Safe copy under lock to prevent tiling/concurrency issues
                frame_to_use = last_frame.copy()
        
        if frame_to_use is None:
            time.sleep(0.1)
            continue
            
        # Encode OUTSIDE the lock to keep the frame-rate high and prevent stalling
        ret, buffer = cv2.imencode('.jpg', frame_to_use, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            time.sleep(0.01)
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # High Speed Stream (~100 FPS Max)
        time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/toggle_live', methods=['POST'])
def toggle_live():
    global live_mode_active
    data = request.json
    live_mode_active = data.get('active', True)
    status_str = "RESUMED" if live_mode_active else "PAUSED"
    print(f"[System] Live Monitoring {status_str} to save resources.")
    return jsonify({"status": "success", "live_active": live_mode_active})

@app.route('/api/list_cameras')
def list_cameras():
    """Returns detected hardware and demo sources."""
    try:
        # [CRITICAL FIX] Return the startup cached cameras!
        # Repeatedly polling cv2.VideoCapture(DSHOW) concurrently crashes the backend.
        return jsonify(DETECTED_CAMERAS)
    except Exception as e:
        return jsonify([{"id": VIDEO_SOURCE, "name": "Default Stream"}]), 200

@app.route('/api/reset_camera', methods=['POST'])
def reset_camera():
    """Switch the live VIDEO_SOURCE and restart capture."""
    global VIDEO_SOURCE, live_thread_token
    data = request.json or {}
    new_id = data.get('index') # Standardizing on 'index' key from UI
    
    if new_id is not None:
        # Convert to int if it's a numeric index
        try:
            if str(new_id).isdigit():
                new_id = int(new_id)
        except Exception as e: 
            print(f"Warning: Could not parse camera ID numeric value: {e}")
        
        print(f"[System] Switching Live Source to: {new_id}")
        
        # Increment token - this effectively kills old threads
        live_thread_token += 1
        
        # Update source
        VIDEO_SOURCE = new_id
        
        # Reset processor for fresh start
        processor.reset()
        
        # Start new thread with new token
        threading.Thread(target=background_capture, args=(live_thread_token,), daemon=True).start()
        return jsonify({"status": "success", "new_source": str(new_id)})
    
    return jsonify({"status": "error", "message": "No index provided"}), 400

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
                    # Elite Preprocessing: Deskew + Pad
                    plate_crop = deskew_plate(plate_crop)
                    plate_crop = apply_padding(plate_crop, pad=15)
                    
                    # [STRICT] Multi-Pass OCR for static image check
                    versions = apply_professional_restoration(plate_crop)
                    all_results = []
                    
                    for v_idx, v_img in enumerate(versions):
                        with processor.lock:
                            res = processor.ocr.ocr(v_img, cls=True)
                        if res and res[0]:
                            lines = [line[1] for idx in range(len(res)) for line in res[idx]]
                            raw_t = "".join([l[0] for l in lines])
                            conf_val = sum([l[1] for l in lines]) / len(lines) if lines else 0
                            
                            # Weight format matches higher
                            clean_t = extract_indian_number_plate([raw_t])
                            if clean_t != "UNKNOWN":
                                all_results.append((clean_t, conf_val + 0.2)) # Priority to strict format
                                raw_texts.append(f"Pass {chr(65+v_idx)}: {clean_t} (STRICT)")
                            else:
                                clean_l = re.sub(r'[^A-Z0-9]', '', raw_t.upper())
                                all_results.append((clean_l, conf_val))
                                raw_texts.append(f"Pass {chr(65+v_idx)}: {clean_l}")

                    if all_results:
                        # Perform voting even on a single image's multiple passes
                        plate_text = processor.character_voting(all_results)
                        # Final strict check
                        plate_text = extract_indian_number_plate([plate_text])
                        
                    ocr_time = (time.time() - start_ocr) * 1000 # ms
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
