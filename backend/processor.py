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
    
    # 1. Bicubic Super-Resolution (Upscale to target 200px height)
    h, w = image.shape[:2]
    scale = 200 / h if h < 200 else 1.0
    upscaled = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    
    # 2. NLMeans Denoising (Preserves edges while removing digital artifacts)
    denoised = cv2.fastNlMeansDenoisingColored(upscaled, None, 10, 10, 7, 21)
    
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

    return [pass_a, pass_b, pass_c, pass_d, pass_e, pass_f]

def character_voting(plate_results):
    """
    Performs character-level voting across multiple plate strings with confidence weighting.
    plate_results: List of (text, confidence) tuples
    """
    if not plate_results:
        return "UNKNOWN"
    
    # Filter out UNKNOWN or very short results
    valid_results = [(s.upper(), conf) for s, conf in plate_results if s != "UNKNOWN" and len(s) >= 4]
    if not valid_results:
        return "UNKNOWN"
    
    # Find the most frequent length
    lengths = [len(s) for s, _ in valid_results]
    target_len = Counter(lengths).most_common(1)[0][0]
    
    # Filter strings to target length (or close to it)
    voters = [(s, conf) for s, conf in valid_results if abs(len(s) - target_len) <= 1]
    
    final_plate = ""
    for i in range(target_len):
        char_weights = {} # {char: total_confidence}
        for s, conf in voters:
            if i < len(s):
                char = s[i]
                char_weights[char] = char_weights.get(char, 0) + conf
        
        if char_weights:
            # Pick character with highest combined confidence
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
    """
    text = " ".join(text_list).upper()
    text = re.sub(r'\bIND\b|\bND\b|\s+', '', text)
    text = re.sub(r'[^A-Z0-9]', '', text)

    # State Codes for Validation
    STATE_CODES = {
        'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DL', 'GA', 'GJ', 'HR', 'HP', 'JH', 
        'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PB', 'RJ', 'SK', 
        'TN', 'TS', 'TR', 'UP', 'UK', 'WB', 'AN', 'DD', 'DN', 'LD', 'PY', 'JK', 'LA'
    }

    # 1. [STRICT] Standard Modern (e.g., AP 39 UX 8273)
    # Pattern: [2 Alpha][2 Numeric][1-2 Alpha][4 Numeric]
    # Allow optional spaces between components
    match = re.search(r'([A-Z]{2})\s*(\d{2})\s*([A-Z]{1,2})\s*(\d{4})', text)
    if match:
        state, rto, ser, num = match.groups()
        if state in STATE_CODES:
            candidate = f"{state}{rto}{ser}{num}"
            return contextual_correction(candidate, "standard")

    # 2. [STRICT] Bharat Series (e.g., 22BH1234AA)
    # Pattern: [2 Numeric][BH][4 Numeric][2 Alpha]
    match = re.search(r'(\d{2})BH(\d{4})([A-Z]{2})', text)
    if match:
        return contextual_correction(match.group(), "bh")

    # 3. [LOOSE] FALLBACK (State Code First + Progressive Filtering)
    for code in STATE_CODES:
        if code in text:
            # Look for State + RTO + Anything + 4 Numbers
            match = re.search(rf'({code})\s*(\d{{1,2}})\s*([A-Z]*)\s*(\d{{4}})', text)
            if match:
                state, rto, ser, num = match.groups()
                # Re-verify and correct
                candidate = contextual_correction(f"{state}{rto}{ser}{num}", "standard")
                if len(candidate) >= 7: return candidate

    # 4. Global Alphanumeric Check (Strict 10 - Standard Candidate)
    # Pattern: Look for 2 Alpha + 2 Num + Anything + 4 Num anywhere
    match = re.search(r'([A-Z]{2})\s*(\d{2})\s*([A-Z0-9]{1,2})\s*(\d{4})', text)
    if match:
        state, rto, ser, num = match.groups()
        if state in STATE_CODES:
            return contextual_correction(f"{state}{rto}{ser}{num}", "standard")

    # 5. [LAST RESORT] Alphanumeric Filter (Min 7, Max 10)
    # Only return if it actually looks like a plate (at least 2 letters, 3 digits)
    candidate = re.sub(r'[^A-Z0-9]', '', text)
    if 7 <= len(candidate) <= 10:
        alpha_count = sum(1 for c in candidate if c.isalpha())
        digit_count = sum(1 for c in candidate if c.isdigit())
        if alpha_count >= 2 and digit_count >= 3:
            return candidate

    return "UNKNOWN"

class BusProcessor:
    def __init__(self, bus_model_path=None, plate_model_path=None, line_position=0.5, line_direction='vertical'):
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
        
        print("Initializing PaddleOCR...")
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True, 
                lang='en', 
                ir_optim=False, 
                enable_mkldnn=False,
                use_tensorrt=False,
                show_log=False
            )
        except Exception as e:
            print(f"Error initializing PaddleOCR: {e}")
            self.ocr = None
        
        self.system_status = "System Ready"
        
        # State tracking for File Pipeline
        self.lock = threading.Lock() 
        
        # State tracking for File Pipeline
        self.tracking_history = {} # {id: last_pos}
        self.processed_ids = {} # {id: "PLATE_NUMBER" or "PENDING"}
        self.proximity_states = {} # {id: {"crossed": bool, "frames": [], "best_size": 0}}
        
        # Proximity settings
        # [REFINED] TRIGGER FOR FRONT PART: 
        # 0.005 triggers earlier (farther) to catch the front clearly as it approaches.
        self.proximity_threshold_ratio = 0.005 
        self.capture_count = 10 # Reduced from 20 for faster high-quality processing
        self.current_session = "live"

    def set_session(self, name):
        """Sets a new storage session name (for logging context only)."""
        # Sanitize name
        clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
        self.current_session = clean_name
        return clean_name
        

    def extract_indian_number_plate(self, text_list):
        return extract_indian_number_plate(text_list)

    def process_frame(self, frame):
        """
        Refactored Frame Processing with PERFORMANCE OPTIMIZATIONS:
        1. Downscale frame for fast YOLO inference (Max 640px)
        2. Tracking with ByteTrack
        3. Crossing Detection (Virtual Line)
        4. Trigger Capture & Analysis
        """
        h_orig, w_orig = frame.shape[:2]
        
        # Performance: Downscale frame for inference (320px is extremely fast on CPU)
        inf_w = 320
        scale = inf_w / w_orig
        inf_h = int(h_orig * scale)
        inf_frame = cv2.resize(frame, (inf_w, inf_h), interpolation=cv2.INTER_LINEAR)

        with self.lock:
            # Detect and Track Buses (320px imgsz provides ~4x speedup over 640px)
            results = self.bus_model.track(inf_frame, persist=True, classes=[5], conf=0.35, tracker="bytetrack.yaml", verbose=False, imgsz=320)
        
        annotated_frame = frame.copy()
        
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
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id_raw, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id_raw) # Ensure it's a python int
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                
                if track_id in self.processed_ids:
                    # [OPTIMIZATION] Already processed or being processed
                    plate = self.processed_ids[track_id]
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
                        
                        # Start or continue capture
                        if len(state["frames"]) < self.capture_count:
                            state["frames"].append(frame.copy())
                            cv2.circle(annotated_frame, (x1+15, y1+15), 8, (0, 0, 255), -1)
                        
                        if len(state["frames"]) == self.capture_count:
                            state["analyzed"] = True
                            self.processed_ids[track_id] = "PENDING"
                            
                            print(f"[AI] >>> ANALYSIS TRIGGERED for Bus {track_id}")
                            thread = threading.Thread(target=self._background_burst_analysis, 
                                                     args=(state["frames"], "DETECTED", track_id))
                            thread.daemon = True
                            thread.start()

                    # Update history for next frame
                    self.tracking_history[track_id] = centroid_x if self.line_direction == 'vertical' else centroid_y

        return annotated_frame

    def _background_burst_analysis(self, burst_frames, direction, track_id):
        """
        Processes captured frames with intelligence:
        1. Localize plate in ALL frames.
        2. Sort by sharpness and pick the top 8 clearest crops.
        3. Perform deskewing and bilateral filtering.
        4. Run OCR and vote.
        """
        plate_candidates = []
        
        for frame in burst_frames:
            with self.lock:
                plate_results = self.plate_model.predict(frame, conf=0.3, verbose=False)
            
            if len(plate_results[0].boxes) > 0:
                box = plate_results[0].boxes.xyxy[0].cpu().numpy()
                px1, py1, px2, py2 = map(int, box)
                plate_crop = frame[py1:py2, px1:px2]
                
                if plate_crop.size > 0:
                    sharpness = calculate_sharpness(plate_crop)
                    plate_candidates.append({"img": plate_crop, "sharpness": sharpness})

        if plate_candidates:
            # Sort by sharpness (Descending) and take top 8
            plate_candidates.sort(key=lambda x: x["sharpness"], reverse=True)
            top_crops = [x["img"] for x in plate_candidates[:8]]
            
            # Additional Preprocessing: Deskewing
            deskewed_crops = [deskew_plate(img) for img in top_crops]
            
            self.run_multi_ocr(deskewed_crops, direction, track_id)
        else:
            print(f"[AI] No plate localized in burst for Bus {track_id}.")
            log_event("numberplate missed", "DETECTED", source=self.current_session, bus_id=track_id)

    def run_multi_ocr(self, plate_images, direction, track_id):
        if self.ocr is None: return

        results = [] # Store (text, confidence)
        
        # [Smart Multi-Pass] Step 1: High-Speed Pass (Grayscale)
        for img in plate_images:
            # Elite Preprocessing: Deskew + Pad
            img = deskew_plate(img)
            img = apply_padding(img, pad=15)
            
            h, w = img.shape[:2]
            scale = 200 / h if h < 200 else 1.0
            prepped = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            prepped = cv2.fastNlMeansDenoisingColored(prepped, None, 10, 10, 7, 21)
            gray = cv2.cvtColor(prepped, cv2.COLOR_BGR2GRAY)
            
            try:
                with self.lock:
                    result = self.ocr.ocr(gray, cls=True)
                if result and result[0]:
                    lines = [line[1] for idx in range(len(result)) for line in result[idx]]
                    raw_text = "".join([l[0] for l in lines])
                    conf = sum([l[1] for l in lines]) / len(lines) if lines else 0
                    clean = extract_indian_number_plate([raw_text])
                    if clean != "UNKNOWN":
                        # Weighted +20% for exact Indian Format match
                        results.append((clean, conf + 0.2))
                    else:
                        clean_loose = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                        if len(clean_loose) >= 4: results.append((clean_loose, conf))
            except Exception as e:
                pass # Suppress OCR errors to prevent spam, but log in debug if needed

        # Step 2: Intelligent Fallback with Advanced Passes (B, C, D)
        # Only run if we don't have a high-confidence match yet
        if not results or max([r[1] for r in results]) < 0.8:
            # Pick the top 5 sharpest frames for heavy processing
            for img in plate_images[:5]:
                # Elite Preprocessing
                img = deskew_plate(img)
                img = apply_padding(img, pad=15)
                
                versions = apply_professional_restoration(img)
                # Pass B (Otsu), Pass C (Adaptive), Pass D (Bilateral HD)
                for processed_img in versions[1:]: 
                    try:
                        with self.lock:
                            result = self.ocr.ocr(processed_img, cls=True)
                        if result and result[0]:
                            lines = [line[1] for idx in range(len(result)) for line in result[idx]]
                            raw_text = "".join([l[0] for l in lines])
                            conf = sum([l[1] for l in lines]) / len(lines) if lines else 0
                            clean = extract_indian_number_plate([raw_text])
                            if clean != "UNKNOWN":
                                results.append((clean, conf + 0.2))
                            else:
                                clean_loose = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                                if len(clean_loose) >= 4: results.append((clean_loose, conf))
                    except Exception as e:
                        pass # Suppress OCR errors in advanced passes

        final_plate_raw = character_voting(results)
        final_plate = extract_indian_number_plate([final_plate_raw])
        
        if (final_plate == "UNKNOWN" or len(final_plate) < 4) and results:
            best_single = max(results, key=lambda x: x[1])
            if best_single[1] > 0.6:
                final_plate = extract_indian_number_plate([best_single[0]])
        
        # Update shared state with the final result
        with self.lock:
            self.processed_ids[track_id] = final_plate
        
        if final_plate != 'UNKNOWN':
            log_event(final_plate, direction, source=self.current_session, bus_id=track_id)
        else:
            # If all 5 frames failed to produce a valid OCR result
            log_event('numberplate missed', direction, source=self.current_session, bus_id=track_id)

    def run_ocr(self, plate_img, direction):
        """Legacy support for single plate processing."""
        self.run_multi_ocr([plate_img], 'DETECTED')

    def get_status(self):
        return self.system_status

    def reset(self):
        self.system_status = "System Ready"
        self.tracking_history = {}
        self.processed_ids = {}
        self.proximity_states = {}
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

        print(f"[VideoMode] Starting analysis of {video_path} (Unified Proximity Logic)")
        
        # Performance mode: process 1 frame every N frames
        skip_frames = 2 # Processes ~10 FPS from 30 FPS video, drastically reducing lag
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            
            # Subsample frames to speed up processing
            if frame_idx % skip_frames != 0:
                continue
            
            # Use the core logic from BusProcessor
            # This handles detection, tracking, crossing, and proximity-based OCR triggering
            annotated_frame = self.processor.process_frame(frame)

            if frame_callback:
                frame_callback(annotated_frame)

            if frame_idx % 20 == 0 and progress_callback:
                progress_callback(int((frame_idx / total_frames) * 100))
            
            # Yield minimally (1ms) to keep UI alive while burning through frames
            time.sleep(0.001) 

        cap.release()
        print(f"[VideoMode] Finished analysis of {video_path}")

    def _extract_plate(self, text_list):
        return extract_indian_number_plate(text_list)
