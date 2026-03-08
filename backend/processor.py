import os
# AGGRESSIVELY disable oneDNN/MKLDNN to prevent "OneDnnContext does not have the input Filter" crash on Windows
os.environ['PADDLE_ONEDNN_DISABLE'] = '1'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_enable_mkldnn_bfloat16'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from database import log_event
import time
import threading
import re
import gc
import queue
from collections import Counter

def calculate_sharpness(image):
    """Calculates the sharpness of an image using Laplacian variance."""
    if image is None or image.size == 0:
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def deskew_plate(image):
    """
    Straightens the license plate using Canny + Hough Transform.
    """
    if image is None or image.size == 0: return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < angle < 45: angles.append(angle)
        
        if angles:
            median_angle = np.median(angles)
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
    return image

def apply_padding(image, pad=10):
    """
    Adds a neutral border to help OCR engines recognize edge characters better.
    """
    if image is None or image.size == 0: return image
    return cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

def apply_morphology(image):
    """Bridge character gaps using morphological closing."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def adjust_gamma(image, gamma=1.0):
    """Adjusts the gamma (brightness/contrast) of an image."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_professional_restoration(image):
    """
    Elite Image Signal Processing (ISP) pipeline for ANPR.
    Returns 4 versions for Multi-Pass OCR.
    """
    if image is None or image.size == 0:
        return [image]
    
    # 1. Lanczos4 Super-Resolution (Upscale to target 240px height for higher detail)
    h, w = image.shape[:2]
    scale = 240 / h if h < 240 else 1.0
    upscaled = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    
    # 2. NLMeans Denoising + Median Blur (Filter salt-and-pepper noise)
    denoised = cv2.fastNlMeansDenoisingColored(upscaled, None, 10, 10, 7, 21)
    denoised = cv2.medianBlur(denoised, 3) # Remove fine digital noise
    
    # 3. CLAHE (Local Contrast Enhancement)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 4. Unsharp Masking (Sharpening)
    gaussian_blur = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian_blur, -0.5, 0)
    
    # Conversion to Grayscale for final passes
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    
    # Pass A: Normalized Grayscale
    pass_a = gray.copy()
    
    # Pass B: Otsu Binarization + Morphology
    _, pass_b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Auto-Invert if needed
    if cv2.countNonZero(pass_b) < (pass_b.shape[0] * pass_b.shape[1] * 0.5):
        pass_b = cv2.bitwise_not(pass_b)
    pass_b = apply_morphology(pass_b)
    
    # Pass C: Adaptive Thresholding
    pass_c = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    if cv2.countNonZero(pass_c) < (pass_c.shape[0] * pass_c.shape[1] * 0.5):
        pass_c = cv2.bitwise_not(pass_c)

    # Pass D: [NEW] High-Definition Detail Preservation
    # Bilateral filter for noise reduction while keeping edges sharp
    pass_d = cv2.bilateralFilter(gray, 9, 75, 75)
    # High-pass filter for extreme edge detection
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    pass_d = cv2.filter2D(pass_d, -1, kernel)

    # Pass E: Gamma Correction for Dark/Night plates
    pass_e = adjust_gamma(gray, gamma=1.5)
    
    # Pass F: Gamma Correction for Washed out/Overexposed plates
    pass_f = adjust_gamma(gray, gamma=0.6)

    # [NEW] Pass G: CLAHE Extreme (for very low contrast)
    clahe_high = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
    pass_g = clahe_high.apply(gray)
    
    # [NEW] Pass H: Sharp Contrast (High Gamma + CLAHE)
    pass_h = adjust_gamma(pass_g, gamma=1.2)

    return [pass_a, pass_b, pass_c, pass_d, pass_e, pass_f, pass_g, pass_h]


def perspective_warp_plate(image):
    """
    Elite Dynamic Warp: Straightens angled plates using perspective transformation.
    Essential for roadside cameras that see buses from the side.
    """
    if image is None or image.size == 0: return image
    
    # 1. Image Preprocessing for contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Edge detection + Contour finding
    edged = cv2.Canny(blurred, 50, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        return deskew_plate(image) # Fallback to simple rotation

    # Find the largest rectangular-ish contour (the plate boundary)
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # If we found 4 corners, we can do a high-precision warp
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        # Order points: TL, TR, BR, BL
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Define destination: standard 350x100 plate aspect ratio
        dst = np.array([
            [0, 0],
            [350 - 1, 0],
            [350 - 1, 100 - 1],
            [0, 100 - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (350, 100))
        return warped
        
    return deskew_plate(image) # Fallback

def character_voting(valid_results):
    """
    WEIGHTED CONFIDENCE VOTING:
    Professional temporal aggregation that considers the quality of each OCR pass.
    valid_results: List of (text, confidence)
    """
    if not valid_results:
        return "UNKNOWN"
    
    # 1. Length Voting: Find most likely plate length
    lengths = [len(s) for s, _ in valid_results]
    target_len = Counter(lengths).most_common(1)[0][0]
    
    # 2. Filter strings to target length (or close to it)
    voters = [(s, weight) for s, weight in valid_results if abs(len(s) - target_len) <= 1]
    
    final_plate = ""
    for i in range(target_len):
        char_weights = {} # {char: total_weight}
        for s, weight in voters:
            if i < len(s):
                char = s[i]
                # Apply an exponential boost to high-confidence characters
                # This ensures a 95% reading heavily outweighs several 40% readings
                effective_weight = np.exp(weight * 3) # Boosted power for clarity
                char_weights[char] = char_weights.get(char, 0) + effective_weight
        
        if char_weights:
            final_char = max(char_weights, key=char_weights.get)
            final_plate += final_char
            
    return final_plate

def contextual_correction(text, pattern_type="standard"):
    """
    Intelligent character correction based on Indian plate rules.
    - pattern_type: "standard" (XX NN XX NNNN) or "bh" (NN BH NNNN XX)
    """
    chars = list(text.upper())
    
    # Standard Index Mapping: 01 23 45 6789
    # Alpha Indices: 0,1, 4,5
    # Numeric Indices: 2,3, 6,7,8,9
    
    alpha_indices = [0, 1, 4, 5] if pattern_type == "standard" else [2, 3, 8, 9]
    numeric_indices = [2, 3, 6, 7, 10, 11] if pattern_type == "standard" else [0, 1, 4, 5, 6, 7]
    # Adjust for variable lengths if needed
    
    correct_map = {
        'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8', 'T': '7'
    }
    inverse_map = {v: k for k, v in correct_map.items()}

    # Patterns vary: Standard (AP39UX8273) or Bharat (22BH1234AA)
    if pattern_type == "standard":
        # XX NN XX NNNN
        # Alpha: 0, 1, (4), (5)
        # Numeric: 2, 3, (6), (7), (8), (9)
        # We handle variable length by checking relative positions
        for i in range(len(chars)):
            # State code (First 2)
            if i < 2 and chars[i] in inverse_map: chars[i] = inverse_map[chars[i]]
            # District code (Next 2)
            elif 1 < i < 4 and chars[i] in correct_map: chars[i] = correct_map[chars[i]]
            # Registration Number (Last 4)
            elif i >= (len(chars) - 4) and chars[i] in correct_map: chars[i] = correct_map[chars[i]]
    
    elif pattern_type == "bh":
        # NN BH NNNN XX
        for i in range(len(chars)):
            if i < 2 and chars[i] in correct_map: chars[i] = correct_map[chars[i]]
            elif 4 < i < 8 and chars[i] in correct_map: chars[i] = correct_map[chars[i]]
            elif i >= (len(chars) - 2) and chars[i] in inverse_map: chars[i] = inverse_map[chars[i]]

    return "".join(chars)

def extract_indian_number_plate(text_list):
    """
    STRICT Indian Number Plate extraction logic.
    Follows MoRTH / HSRP guidelines (Standard & BH Series).
    Checks for 30+ potential misreads with aggressive regex.
    """
    text = " ".join(text_list).upper()
    # Remove common artifacts like 'IND', 'NOT', or spaces
    text = re.sub(r'[^A-Z0-9]', '', text)
    text = text.replace("IND", "")

    # State Codes for Validation
    STATE_CODES = {
        'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DL', 'GA', 'GJ', 'HR', 'HP', 'JH', 
        'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PB', 'RJ', 'SK', 
        'TN', 'TS', 'TR', 'UP', 'UK', 'WB', 'AN', 'DD', 'DN', 'LD', 'PY', 'JK', 'LA'
    }

    # 1. [STRICT] Standard (e.g., AP 38 BP 1234)
    standard_match = re.search(r'([A-Z]{2})(\d{1,2})([A-Z]{1,2})(\d{4})', text)
    if standard_match:
        state, district, series, num = standard_match.groups()
        if state in STATE_CODES:
            # Pad district if single digit
            if len(district) == 1: district = "0" + district
            candidate = f"{state}{district}{series}{num}"
            return contextual_correction(candidate, "standard")

    # 2. [STRICT] BH Series (e.g., 22BH1234AA)
    bh_match = re.search(r'(\d{2})BH(\d{4})([A-Z]{2})', text)
    if bh_match:
        return contextual_correction(bh_match.group(), "bh")

    # 3. [LOOSE] State Code Fallback
    for code in STATE_CODES:
        if code in text:
            idx = text.find(code)
            candidate = text[idx:idx+15] # Grab enough characters
            match = re.search(rf'({code})(\d{{2}})([A-Z]{{1,2}})(\d{{4}})', candidate)
            if match:
                return contextual_correction(match.group(), "standard")

    # 4. [EXTREME RESCUE] Filter Alpha-Num count
    chars = re.sub(r'[^A-Z0-9]', '', text)
    if 8 <= len(chars) <= 10:
        return chars

    return "UNKNOWN"

class BusProcessor:
    def __init__(self, bus_model_path=None, plate_model_path=None, line_position=0.5, line_direction='horizontal'):
        """
        Refactored for Video File Pipeline.
        - line_position: 0.0 to 1.0 (percent of width/height)
        - line_direction: 'vertical' (for left-right) or 'horizontal' (for top-bottom)
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        if bus_model_path is None:
            bus_model_path = os.path.join(base_path, 'yolov8n.pt')
        if plate_model_path is None:
            plate_model_path = os.path.join(base_path, 'best.pt')
            
        # Virtual line configuration
        self.line_position = line_position
        self.line_direction = line_direction
        
        print(f"Loading Models (Bus: {bus_model_path}, Plate: {plate_model_path})...")
        self.bus_model = YOLO(bus_model_path)
        self.bus_model.to('cpu')
        self.plate_model = YOLO(plate_model_path)
        self.plate_model.to('cpu')
        gc.collect()
        
        print("Initializing PaddleOCR (Elite Accuracy Mode)...")
        try:
            # ELITE ACCURACY CONFIG: 
            # - det_db_thresh=0.3: more sensitive to faint text
            # - use_angle_cls=True: handles skewed/leaning plates
            self.ocr = PaddleOCR(
                use_angle_cls=True, 
                lang='en', 
                ir_optim=True, # Improved optimization
                enable_mkldnn=False,
                use_tensorrt=False,
                show_log=False,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                rec_batch_num=6 # Faster batch processing in background
            )
        except Exception as e:
            print(f"Error initializing PaddleOCR: {e}")
            self.ocr = None
        
        self.system_status = "System Ready"
        
        # ELITE v2.0 ARCHITECTURE: Job Queue & Singleton Worker
        self.ocr_queue = queue.Queue(maxsize=500) # Buffer for ~100 buses
        self.worker_thread = threading.Thread(target=self._ocr_worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        # State tracking for File Pipeline
        self.tracking_lock = threading.Lock() # Protects state tracking variables
        self.inference_lock = threading.Lock() # Protects YOLO and OCR model calls
        
        # State tracking for File Pipeline
        self.tracking_history = {} # {id: last_pos}
        self.processed_ids = {} # {id: "PLATE_NUMBER" or "PENDING"}
        self.proximity_states = {} # {id: {"crossed": bool, "frames": [], "best_size": 0}}
        self.active_track_ids = set()
        
        # Proximity settings
        # [REFINED] TRIGGER FOR FRONT PART: 
        # Wait until the bus occupies at least 7% of the frame for clearer plate detail.
        self.proximity_threshold_ratio = 0.07 
        self.capture_count = 5 # Further reduced (from 8 to 5) for faster trigger. Accuracy maintained via selective sampling.
        self.current_session = "live"
        self.last_cleanup_time = time.time()

    def set_session(self, name):
        """Sets a new storage session name (for logging context only)."""
        # Sanitize name
        clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
        self.current_session = clean_name
        return clean_name

    def _ocr_worker_loop(self):
        """
        Background Worker: Consumes bus jobs from the queue.
        Ensures the live feed stays at 30 FPS even during heavy traffic.
        """
        print("[System] OCR Worker Thread Started (Idle Mode)")
        while True:
            try:
                # Wait for a job (block=True)
                job = self.ocr_queue.get(timeout=10)
                burst_frames, direction, track_id = job
                
                print(f"[Worker] Processing Bus {track_id} ({len(burst_frames)} frames in queue)")
                self._background_burst_analysis(burst_frames, direction, track_id)
                
                # Cleanup: Explicitly delete frames from memory
                del burst_frames
                self.ocr_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Worker] Fatal Error: {e}")
                time.sleep(1)

        

    def extract_indian_number_plate(self, text_list):
        return extract_indian_number_plate(text_list)

    def cleanup_stale_data(self):
        """Purges old tracking and proximity data to prevent memory growth over 24/7 operation."""
        now = time.time()
        # Cleanup every 5 minutes
        if now - self.last_cleanup_time < 300:
            return
            
        print("[System] Performing periodic memory cleanup...")
        with self.tracking_lock:
            # Clear tracking history for IDs not seen in 10 minutes
            self.tracking_history.clear() 
            self.proximity_states.clear()
            
            # Keep processed_ids for a bit longer to prevent re-processing the same bus if it lingers
            # but clear if it gets too large
            if len(self.processed_ids) > 1000:
                self.processed_ids.clear()
                
            self.last_cleanup_time = now

    def process_frame(self, frame, imgsz=320):
        """
        Refactored Frame Processing with PERFORMANCE OPTIMIZATIONS:
        1. Downscale frame for fast YOLO inference
        2. Tracking with ByteTrack
        3. Crossing Detection (Virtual Line)
        4. Trigger Capture & Analysis
        """
        # 0. Global Memory Cleanup
        self.cleanup_stale_data()

        h_orig, w_orig = frame.shape[:2]
        
        # Performance: Downscale frame for inference
        inf_w = imgsz
        scale = inf_w / w_orig
        inf_h = int(h_orig * scale)
        inf_frame = cv2.resize(frame, (inf_w, inf_h), interpolation=cv2.INTER_LINEAR)

        with self.inference_lock:
            # Detect and Track Buses (imgsz is now dynamic: 320 for Live, 640 for Video)
            results = self.bus_model.track(inf_frame, persist=True, classes=[5], conf=0.35, tracker="bytetrack.yaml", verbose=False, imgsz=imgsz)
        
        annotated_frame = frame.copy()
        current_active_ids = set()
        
        # Draw Virtual Gate Line
        if self.line_direction == 'vertical':
            line_x = int(w_orig * self.line_position)
            cv2.line(annotated_frame, (line_x, 0), (line_x, h_orig), (255, 0, 0), 2)
            cv2.putText(annotated_frame, "VIRTUAL LINE", (line_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            line_y = int(h_orig * self.line_position)
            cv2.line(annotated_frame, (0, line_y), (w_orig, line_y), (255, 0, 0), 2)
            cv2.putText(annotated_frame, "VIRTUAL LINE", (10, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Scale boxes back to original coordinates
            boxes = results[0].boxes.xyxy.cpu().numpy() / scale
            track_ids = results[0].boxes.id.int().cpu().numpy()
            current_active_ids = set(track_ids.tolist())
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id_raw, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id_raw) # Ensure it's a python int
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                
                with self.tracking_lock:
                    is_already_processed = track_id in self.processed_ids
                    if is_already_processed:
                        plate = self.processed_ids[track_id]
                
                if is_already_processed:
                    # [OPTIMIZATION] Already processed or being processed
                    color = (255, 255, 0) if plate == "PENDING" else (0, 255, 255) # Cyan for Completed
                    label = f"Bus {track_id}: {plate}" if plate != "PENDING" else f"Bus {track_id}: ANALYZING..."
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Skip all proximity/capture logic for this bus to save CPU
                    continue

                # Default Drawing for Pending Buses
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Bus {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 1. Crossing Detection
                with self.tracking_lock:
                    # Initialize state for new IDs
                    if track_id not in self.proximity_states:
                        self.proximity_states[track_id] = {"crossed": False, "frames": [], "analyzed": False}
                    
                    state = self.proximity_states[track_id]
                    
                    # Detect Crossing
                    if track_id in self.tracking_history:
                        prev_pos = self.tracking_history[track_id]
                        line_val = (w_orig * self.line_position) if self.line_direction == 'vertical' else (h_orig * self.line_position)
                        curr_pos = centroid_x if self.line_direction == 'vertical' else centroid_y
                        
                        if (prev_pos < line_val <= curr_pos) or (prev_pos > line_val >= curr_pos):
                            state["crossed"] = True
                            print(f"[AI] Bus {track_id} crossed virtual line.")

                    # 2. Trigger Burst Capture (Once crossed AND near enough OR if already very large)
                    area = (x2 - x1) * (y2 - y1)
                    area_ratio = area / (w_orig * h_orig)

                    # Trigger if crossed OR if bus is already very large (fallback for videos starting mid-way)
                    should_trigger = state["crossed"] or area_ratio > 0.15

                    if should_trigger and not state["analyzed"]:
                        if area_ratio > self.proximity_threshold_ratio or len(state["frames"]) > 0:
                            if len(state["frames"]) == 0:
                                print(f"[AI] >>> START CAPTURE for Bus {track_id} (Ratio: {area_ratio:.3f})")
                            
                            # Start or continue capture with selective sampling (every other frame)
                            # This ensures we get different angles rather than 5 identical blurry frames
                            if len(state["frames"]) < self.capture_count:
                                if state.get("frame_index", 0) % 2 == 0: 
                                    # ELITE UPGRADE: Bumper-Only Selective Cropping
                                    bw = x2 - x1
                                    bh = y2 - y1
                                    
                                    # Logic: Focus on the "Bumper Belt" (middle-to-bottom portion of the bus)
                                    buffer_y1 = y1 + int(bh * 0.4)
                                    buffer_y2 = y1 + int(bh * 0.9) 
                                    
                                    bumper_crop = frame[buffer_y1:buffer_y2, x1:x2]
                                    if bumper_crop.size > 0:
                                        state["frames"].append({
                                            "img": bumper_crop.copy(), 
                                            "full_box": (x1, y1, x2, y2),
                                            "crop_offset": (x1, buffer_y1)
                                        })
                                        # Visual Feedback: Red dot for capture
                                        cv2.circle(annotated_frame, (x1+15, y1+15), 8, (0, 0, 255), -1)
                            
                            if len(state["frames"]) >= self.capture_count and not state["analyzed"]:
                                state["analyzed"] = True
                                self.processed_ids[track_id] = "ANALYZING..."
                                
                                # ELITE v2.0: Push to Queue instead of starting new thread
                                bus_job = (list(state["frames"]), "DETECTED", track_id)
                                try:
                                    self.ocr_queue.put_nowait(bus_job)
                                    print(f"[AI] >>> QUEUED Bus {track_id} for Deferred OCR.")
                                except queue.Full:
                                    print(f"[AI] Warning: Job Queue Full! Dropping Bus {track_id}")

                    # Update history for next frame
                    self.tracking_history[track_id] = centroid_x if self.line_direction == 'vertical' else centroid_y

        with self.tracking_lock:
            self.active_track_ids = current_active_ids
        return annotated_frame

    def _background_burst_analysis(self, burst_frames, direction, track_id):
        """
        Processes captured frames with intelligence:
        1. Localize plate in ALL frames.
        2. Sort by sharpness and pick the top 5 clearest crops.
        3. Perform deskewing and bilateral filtering.
        4. Run OCR and vote.
        """
        plate_candidates = []
        
        for frame_data in burst_frames:
            bus_crop = frame_data["img"]
            if bus_crop.size == 0: continue

            with self.inference_lock:
                # Run plate model directly on the bumper crop
                plate_results = self.plate_model.predict(bus_crop, conf=0.15, verbose=False)
            
            if len(plate_results[0].boxes) > 0:
                box = plate_results[0].boxes.xyxy[0].cpu().numpy()
                px1, py1, px2, py2 = map(int, box)
                
                # [ELITE v2.0] LOOSE PADDING (25% Expansion)
                # This is the industry-standard fix for clipped characters
                pw = px2 - px1
                ph = py2 - py1
                pad_w = int(pw * 0.25)
                pad_h = int(ph * 0.25)
                
                px1 = max(0, px1 - pad_w)
                py1 = max(0, py1 - pad_h)
                px2 = min(bus_crop.shape[1], px2 + pad_w)
                py2 = min(bus_crop.shape[0], py2 + pad_h)
                
                plate_crop = bus_crop[py1:py2, px1:px2]
                
                if plate_crop.size > 0:
                    sharpness = calculate_sharpness(plate_crop)
                    plate_candidates.append({"img": plate_crop, "sharpness": sharpness})

        if plate_candidates:
            # Sort by sharpness (Descending) and take top 5 (matches capture_count)
            plate_candidates.sort(key=lambda x: x["sharpness"], reverse=True)
            top_crops = [x["img"] for x in plate_candidates[:5]]
            
            # Elite Upgrade: Use Perspective Warping instead of simple deskewing
            # This is essential for plates seen at an angle
            warped_crops = [perspective_warp_plate(img) for img in top_crops]
            
            self.run_multi_ocr(warped_crops, direction, track_id)
        else:
            print(f"[AI] No plate localized in burst for Bus {track_id}.")
            log_event("numberplate missed", "DETECTED", source=self.current_session, bus_id=track_id)
            with self.tracking_lock:
                self.processed_ids[track_id] = "UNKNOWN"

    def run_multi_ocr(self, plate_images, direction, track_id):
        if self.ocr is None: return

        ocr_raw_data = [] # Store (text, confidence)
        standard_regex = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$')
        
        # [Elite Multi-Pass] Process each warped image
        for img in plate_images:
            # 1. Advanced ISP: Contrast Enhancement (CLAHE)
            # Essential for readable HSRP plates under bright lights or shadows
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # 2. Adaptive Gamma: Auto-correct for Shadows or Harsh Sunlight
            gray_crop = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray_crop)
            
            # Target average brightness is ~120
            if avg_brightness < 80:
                # Under shadows - boost brightness
                enhanced = adjust_gamma(enhanced, gamma=1.5)
            elif avg_brightness > 180:
                # Direct sunlight/glare - reduce contrast
                enhanced = adjust_gamma(enhanced, gamma=0.7)

            # 3. Resizing & Filtering
            h, w = enhanced.shape[:2]
            scale = 240 / h if h < 240 else 1.0
            prepped = cv2.resize(enhanced, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            prepped = cv2.bilateralFilter(prepped, 5, 75, 75)
            
            try:
                res = self.ocr.ocr(prepped, cls=True)
                if res and res[0]:
                    for line in res[0]:
                        text, conf = line[1]
                        text_clean = text.upper().replace(" ", "")
                        if len(text_clean) >= 4:
                            # Apply validation bonus
                            weight = conf
                            if standard_regex.match(text_clean):
                                weight += 0.5
                            ocr_raw_data.append((text_clean, weight))
            except Exception as e:
                print(f"[OCR] Pass Error: {e}")

        # Elite Character Voting (Confidence-Weighted)
        final_plate = character_voting(ocr_raw_data)
        
        # Apply strict contextual corrections
        pattern = "standard" if len(final_plate) >= 8 else "bh"
        final_plate = contextual_correction(final_plate, pattern)

        # Final result is (plate, average_confidence)
        # 3. Dynamic Confidence Rescue [ELITE v2.0]
        # We calculate initial confidence FIRST to decide if we need a rescue
        winner_conf = 0
        if ocr_raw_data and final_plate != "UNKNOWN":
            voters = [w for p, w in ocr_raw_data if p == final_plate]
            if voters: winner_conf = sum(voters) / len(voters)

        # If the result is UNKNOWN or very low confidence, trigger a deep scan
        is_weak = final_plate == "UNKNOWN" or (ocr_raw_data and winner_conf < 0.6)
        
        if is_weak and plate_images:
            # Pick the sharpest frame (first in list after sorting)
            best_img = plate_images[0]
            # Try a "Deep Rescue" on the top 3 specialized filters
            rescue_v = apply_professional_restoration(best_img)
            rescue_results = []
            
            for v_img in rescue_v[:3]: # Normalized, Otsu, and Adaptive filters
                try:
                    res = self.ocr.ocr(v_img, cls=True)
                    if res and res[0]:
                        for line in res[0]:
                            text, conf = line[1]
                            text_clean = extract_indian_number_plate([text])
                            if text_clean != "UNKNOWN":
                                rescue_results.append((text_clean, conf))
                except: continue
            
            if rescue_results:
                # If rescue found a better/standard plate, override
                best_rescue = max(rescue_results, key=lambda x: x[1])
                if final_plate == "UNKNOWN" or best_rescue[1] > (winner_conf + 0.1):
                    final_plate, winner_conf = best_rescue
                    print(f"[AI] >>> RESCUE SUCCESS: {final_plate} (New Conf: {winner_conf:.2f})")

        with self.tracking_lock:
            self.processed_ids[track_id] = final_plate
            print(f"[AI] FINAL RESULT for Bus {track_id}: {final_plate} (Conf: {winner_conf:.2f})")
        
        # Log to DB/CSV with confidence
        log_event((final_plate, round(winner_conf, 2)), direction, source=self.current_session, bus_id=track_id)
    def get_status(self):
        return self.system_status

    def reset(self):
        """Clears all tracking and proximity states for a fresh start."""
        self.system_status = "System Ready"
        with self.tracking_lock:
            self.tracking_history.clear()
            self.processed_ids.clear()
            self.proximity_states.clear()
            self.active_track_ids.clear()
            # Clear queue safely
            while not self.ocr_queue.empty():
                try: self.ocr_queue.get_nowait()
                except: break
            print("[Processor] State Reset for new video loop.")

class VideoUploadProcessor:
    def __init__(self, bus_model_path=None, plate_model_path=None, line_position=0.5, line_direction='horizontal'):
        """
        Unified Video Upload Processor.
        Uses a BusProcessor instance to maintain consistency in logic.
        """
        self.processor = BusProcessor(
            bus_model_path=bus_model_path,
            plate_model_path=plate_model_path,
            line_position=line_position,
            line_direction=line_direction
        )
        self.lock = threading.Lock()

    def process_video(self, video_path, progress_callback=None, frame_callback=None):
        """Processes an uploaded video file using the unified proximity logic."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Error: Could not determine frame count for {video_path}")
            cap.release()
            return
            
        frame_idx = 0
        
        # Set a unique session name for this upload to organize frames in subfolders
        video_name = os.path.basename(video_path).split('.')[0]
        timestamp = int(time.time())
        self.processor.set_session(f"upload_{video_name}_{timestamp}")
        
        # Reset processor state for this specific video run
        self.processor.tracking_history = {}
        self.processor.processed_ids = {} # Unified dictionary format
        self.processor.proximity_states = {}
        
        print(f"[VideoMode] Starting analysis of {video_path} (High Accuracy + Smooth UI Mode)")
        
        # [PERFORMANCE TWEAK] Process every 2nd frame (approx 15 FPS)
        # We need more frames to guarantee we catch the exact moment the plate is sharpest.
        base_skip_frames = 2 
        dynamic_skip_counter = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            
            # [DYNAMIC INTELLIGENT FAST FORWARDING]
            # If we've already captured the front of the bus, skip YOLO frames rapidly
            if dynamic_skip_counter > 0:
                dynamic_skip_counter -= 1
                continue
            
            # Subsample frames to speed up processing
            if frame_idx % base_skip_frames != 0:
                continue
            
            # [MAXIMUM ACCURACY] Use 640px imgsz (up from 480px)
            # We compensate for the slower 640px by skipping more frames above.
            annotated_frame = self.processor.process_frame(frame, imgsz=640)
            
            # Fast-Forward check: If ALL currently visible buses have already been captured (in processed_ids)
            with self.processor.tracking_lock:
                active_ids = self.processor.active_track_ids
                processed_ids = self.processor.processed_ids.copy()
            
            if active_ids and all(tid in processed_ids for tid in active_ids):
                # Skip the next 15 frames rapidly to get past the remainder of the bus
                dynamic_skip_counter = 15
                cv2.putText(annotated_frame, "FAST FORWARDING >>>", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

            if frame_callback:
                frame_callback(annotated_frame)

            if frame_idx % 20 == 0 and progress_callback:
                progress_callback(int((frame_idx / total_frames) * 100))
            
            # [CRITICAL UI FIX] Yield 30ms (up from 1ms)
            # This allows the Flask backend to actually breathe and send the frames to your React frontend,
            # eliminating the "lagging" or frozen UI effect.
            time.sleep(0.03) 

        cap.release()
        print(f"[VideoMode] Finished analysis of {video_path}")

    def _extract_plate(self, text_list):
        return extract_indian_number_plate(text_list)
